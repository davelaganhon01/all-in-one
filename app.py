import json
import logging
import os
import re
import secrets
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import text, update
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from extensions import csrf, db, login_manager
from models import AnalysisJob, UploadSession, User


ALLOWED_EXTENSIONS = {"mp3", "wav"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 256 * 1024
UPLOAD_SESSION_MAX_AGE = timedelta(hours=24)
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,40}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
logger = logging.getLogger(__name__)
ANALYSIS_SLOTS = threading.BoundedSemaphore(2)


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)
    upload_folder = Path(
        os.environ.get("ALLIN1_UPLOAD_FOLDER", instance_path / "uploads")
    ).resolve()
    upload_folder.mkdir(parents=True, exist_ok=True)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        SQLALCHEMY_DATABASE_URI=os.environ.get(
            "DATABASE_URL", f"sqlite:///{instance_path / 'allin1-web.sqlite'}"
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=str(upload_folder),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "").lower()
        in {"1", "true", "yes"},
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    login_manager.login_view = "login"
    login_manager.login_message_category = "info"

    with app.app_context():
        db.create_all()

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return db.session.get(User, int(user_id))
        except (TypeError, ValueError):
            return None

    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not USERNAME_PATTERN.fullmatch(username):
                flash(
                    "Use 3–40 letters, numbers, dots, dashes, or underscores for your username.",
                    "error",
                )
            elif len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
                flash("Enter a valid email address.", "error")
            elif len(password) < 12:
                flash("Your password must be at least 12 characters.", "error")
            else:
                try:
                    # Serialize first-account checks so concurrent registrations
                    # cannot both receive owner privileges in SQLite.
                    db.session.execute(text("BEGIN IMMEDIATE"))
                    if User.query.filter(
                        (User.username == username) | (User.email == email)
                    ).first():
                        db.session.rollback()
                        flash("That username or email is already registered.", "error")
                    else:
                        is_first_user = User.query.count() == 0
                        user = User(
                            username=username,
                            email=email,
                            role="owner" if is_first_user else "user",
                            status="approved" if is_first_user else "pending",
                        )
                        user.set_password(password)
                        db.session.add(user)
                        db.session.commit()
                        if is_first_user:
                            login_user(user)
                            flash("Your owner account is ready.", "success")
                            return redirect(url_for("dashboard"))
                        flash(
                            "Registration received. Your account is pending owner approval.",
                            "success",
                        )
                        return redirect(url_for("login"))
                except Exception:
                    db.session.rollback()
                    logger.exception("Could not register account")
                    flash("We couldn't create your account. Please try again.", "error")

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            identity = request.form.get("identity", "").strip()
            password = request.form.get("password", "")
            user = User.query.filter(
                (User.username == identity) | (User.email == identity.lower())
            ).first()
            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for("dashboard"))
            flash("Incorrect username, email, or password.", "error")

        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        flash("You have been signed out.", "success")
        return redirect(url_for("login"))

    def admin_required(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if not current_user.is_admin:
                abort(403)
            return view(*args, **kwargs)

        return wrapped

    def access_block_reason(user):
        if user.status == "pending":
            return "Your account is waiting for owner approval."
        if user.status == "rejected":
            return "Your account was not approved. Contact the owner for help."
        if user.status == "suspended":
            return "Your account is suspended. Contact the owner for help."
        if user.role == "owner":
            return None
        if user.status != "approved":
            return "Your account is not approved to use the analyzer."
        if user.expires_at is None:
            return "An owner must assign an account expiry before you can analyze audio."
        if user.expires_at <= utcnow():
            return "Your account access has expired. Contact the owner to renew it."
        if user.analysis_limit is None:
            return "An owner must assign an analysis limit before you can analyze audio."
        if user.analysis_count >= user.analysis_limit:
            return "You have used all of your allowed analyses."
        return None

    @app.route("/dashboard", methods=["GET", "POST"])
    @login_required
    def dashboard():
        result = None
        if request.method == "POST":
            reason = access_block_reason(current_user)
            if reason:
                flash(reason, "error")
                return redirect(url_for("dashboard"))

            audio = request.files.get("audio")
            if not audio or not audio.filename:
                flash("Choose an MP3 or WAV file to analyze.", "error")
                return redirect(url_for("dashboard"))

            original_name = secure_filename(audio.filename)
            extension = Path(original_name).suffix.lower().lstrip(".")
            if not original_name or extension not in ALLOWED_EXTENSIONS:
                flash("Only MP3 and WAV audio files are supported.", "error")
                return redirect(url_for("dashboard"))

            stored_path = Path(app.config["UPLOAD_FOLDER"]) / (
                f"{uuid.uuid4().hex}.{extension}"
            )
            try:
                audio.save(stored_path)
                if stored_path.stat().st_size == 0:
                    stored_path.unlink(missing_ok=True)
                    flash("The selected audio file is empty.", "error")
                    return redirect(url_for("dashboard"))

                if current_user.role != "owner":
                    reservation = db.session.execute(
                        update(User)
                        .where(
                            User.id == current_user.id,
                            User.status == "approved",
                            User.expires_at > utcnow(),
                            User.analysis_count < User.analysis_limit,
                        )
                        .values(analysis_count=User.analysis_count + 1)
                    )
                    db.session.commit()
                    if reservation.rowcount != 1:
                        db.session.refresh(current_user)
                        stored_path.unlink(missing_ok=True)
                        flash(
                            access_block_reason(current_user)
                            or "Your analyzer access is unavailable.",
                            "error",
                        )
                        return redirect(url_for("dashboard"))

                job_id = uuid.uuid4().hex
                job = AnalysisJob(
                    id=job_id,
                    user_id=current_user.id,
                    filename=original_name,
                    audio_path=str(stored_path),
                )
                db.session.add(job)
                db.session.commit()
                try:
                    thread = threading.Thread(
                        target=run_analysis_job,
                        args=(app, job_id),
                        name=f"analysis-{job_id[:8]}",
                        daemon=True,
                    )
                    thread.start()
                except Exception:
                    logger.exception("Could not start analysis job %s", job_id)
                    job.status = "failed"
                    job.error_message = (
                        "The analyzer could not be started. Please try again."
                    )
                    db.session.commit()
                return redirect(url_for("analysis_job", job_id=job_id))
            except Exception:
                db.session.rollback()
                logger.exception("Could not queue analysis for user %s", current_user.id)
                flash(
                    "We couldn't queue your analysis. Please try again.",
                    "error",
                )
                return redirect(url_for("dashboard"))

        return render_template(
            "dashboard.html",
            access_reason=access_block_reason(current_user),
            result=result,
        )

    @app.route("/uploads", methods=["POST"])
    @login_required
    def create_upload():
        reason = access_block_reason(current_user)
        if reason:
            return jsonify(error=reason), 403

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Invalid upload request."), 400

        original_name = secure_filename(str(payload.get("filename", "")))
        extension = Path(original_name).suffix.lower().lstrip(".")
        file_size = payload.get("size")
        if not original_name or extension not in ALLOWED_EXTENSIONS:
            return jsonify(error="Only MP3 and WAV audio files are supported."), 400
        if (
            isinstance(file_size, bool)
            or not isinstance(file_size, int)
            or not 0 < file_size <= MAX_UPLOAD_BYTES
        ):
            return jsonify(error="Choose an audio file smaller than 100 MB."), 413

        cutoff = utcnow() - UPLOAD_SESSION_MAX_AGE
        expired_sessions = UploadSession.query.filter(
            UploadSession.created_at < cutoff
        ).all()
        chunk_root = Path(app.config["UPLOAD_FOLDER"]) / ".chunks"
        for expired in expired_sessions:
            shutil.rmtree(chunk_root / expired.id, ignore_errors=True)
            db.session.delete(expired)
        if expired_sessions:
            db.session.commit()

        upload_id = uuid.uuid4().hex
        db.session.add(
            UploadSession(
                id=upload_id,
                user_id=current_user.id,
                filename=original_name,
                file_size=file_size,
            )
        )
        db.session.commit()
        (chunk_root / upload_id).mkdir(parents=True, exist_ok=True)
        return jsonify(
            upload_id=upload_id,
            chunk_size=UPLOAD_CHUNK_BYTES,
            chunk_count=(file_size + UPLOAD_CHUNK_BYTES - 1) // UPLOAD_CHUNK_BYTES,
        )

    @app.route("/uploads/<upload_id>/chunks/<int:index>", methods=["POST"])
    @login_required
    def upload_chunk(upload_id, index):
        upload = db.session.get(UploadSession, upload_id)
        if upload is None:
            return jsonify(error="This upload has expired. Please try again."), 404
        if upload.user_id != current_user.id:
            abort(403)
        reason = access_block_reason(current_user)
        if reason:
            return jsonify(error=reason), 403

        chunk_count = (upload.file_size + UPLOAD_CHUNK_BYTES - 1) // UPLOAD_CHUNK_BYTES
        if index < 0 or index >= chunk_count:
            return jsonify(error="Invalid upload chunk."), 400

        expected_size = min(
            UPLOAD_CHUNK_BYTES, upload.file_size - index * UPLOAD_CHUNK_BYTES
        )
        if request.content_length != expected_size:
            return jsonify(error="The upload chunk has an invalid size."), 400

        chunk_path = (
            Path(app.config["UPLOAD_FOLDER"])
            / ".chunks"
            / upload.id
            / str(index)
        )
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = chunk_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary_path.open("xb") as chunk_file:
                shutil.copyfileobj(request.stream, chunk_file, length=UPLOAD_CHUNK_BYTES)
            if temporary_path.stat().st_size != expected_size:
                return jsonify(error="The upload chunk was incomplete."), 400
            os.replace(temporary_path, chunk_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return jsonify(received=True)

    @app.route("/uploads/<upload_id>", methods=["DELETE"])
    @login_required
    def cancel_upload(upload_id):
        upload = db.session.get(UploadSession, upload_id)
        if upload is None:
            return "", 204
        if upload.user_id != current_user.id:
            abort(403)

        db.session.delete(upload)
        db.session.commit()
        shutil.rmtree(
            Path(app.config["UPLOAD_FOLDER"]) / ".chunks" / upload_id,
            ignore_errors=True,
        )
        return "", 204

    @app.route("/uploads/<upload_id>/complete", methods=["POST"])
    @login_required
    def complete_upload(upload_id):
        upload = db.session.get(UploadSession, upload_id)
        if upload is None:
            return jsonify(error="This upload has expired. Please try again."), 404
        if upload.user_id != current_user.id:
            abort(403)
        reason = access_block_reason(current_user)
        if reason:
            return jsonify(error=reason), 403

        chunk_root = Path(app.config["UPLOAD_FOLDER"]) / ".chunks" / upload.id
        stored_path = Path(app.config["UPLOAD_FOLDER"]) / (
            f"{uuid.uuid4().hex}{Path(upload.filename).suffix.lower()}"
        )
        chunk_count = (upload.file_size + UPLOAD_CHUNK_BYTES - 1) // UPLOAD_CHUNK_BYTES
        try:
            with stored_path.open("xb") as assembled:
                for index in range(chunk_count):
                    chunk_path = chunk_root / str(index)
                    expected_size = min(
                        UPLOAD_CHUNK_BYTES,
                        upload.file_size - index * UPLOAD_CHUNK_BYTES,
                    )
                    if not chunk_path.is_file() or chunk_path.stat().st_size != expected_size:
                        stored_path.unlink(missing_ok=True)
                        return jsonify(error="The upload is incomplete. Please retry it."), 400
                    with chunk_path.open("rb") as chunk_file:
                        shutil.copyfileobj(chunk_file, assembled, length=UPLOAD_CHUNK_BYTES)
            if stored_path.stat().st_size != upload.file_size:
                stored_path.unlink(missing_ok=True)
                return jsonify(error="The uploaded file size did not match."), 400

            if current_user.role != "owner":
                reservation = db.session.execute(
                    update(User)
                    .where(
                        User.id == current_user.id,
                        User.status == "approved",
                        User.expires_at > utcnow(),
                        User.analysis_count < User.analysis_limit,
                    )
                    .values(analysis_count=User.analysis_count + 1)
                )
                if reservation.rowcount != 1:
                    db.session.rollback()
                    stored_path.unlink(missing_ok=True)
                    db.session.refresh(current_user)
                    return jsonify(
                        error=access_block_reason(current_user)
                        or "Your analyzer access is unavailable."
                    ), 403

            job = AnalysisJob(
                id=uuid.uuid4().hex,
                user_id=current_user.id,
                filename=upload.filename,
                audio_path=str(stored_path),
            )
            db.session.add(job)
            db.session.delete(upload)
            db.session.commit()
        except Exception:
            db.session.rollback()
            stored_path.unlink(missing_ok=True)
            raise

        shutil.rmtree(chunk_root, ignore_errors=True)
        try:
            thread = threading.Thread(
                target=run_analysis_job,
                args=(app, job.id),
                name=f"analysis-{job.id[:8]}",
                daemon=True,
            )
            thread.start()
        except Exception:
            logger.exception("Could not start analysis job %s", job.id)
            job.status = "failed"
            job.error_message = "The analyzer could not be started. Please try again."
            db.session.commit()
        return jsonify(redirect=url_for("analysis_job", job_id=job.id))

    def owned_analysis_job(job_id):
        job = db.get_or_404(AnalysisJob, job_id)
        if job.user_id != current_user.id and not current_user.is_admin:
            abort(403)
        return job

    @app.route("/analysis/<job_id>")
    @login_required
    def analysis_job(job_id):
        job = owned_analysis_job(job_id)
        if job.status == "completed":
            return render_template(
                "dashboard.html",
                access_reason=access_block_reason(current_user),
                result=json.loads(job.result_json),
            )
        return render_template("analysis.html", job=job)

    @app.route("/analysis/<job_id>/status")
    @login_required
    def analysis_status(job_id):
        job = owned_analysis_job(job_id)
        return jsonify(status=job.status, error=job.error_message)

    @app.route("/admin")
    @login_required
    def admin_redirect():
        if not current_user.is_admin:
            abort(403)
        return redirect(url_for("admin_dashboard"))

    @app.route("/admin/users")
    @admin_required
    def admin_dashboard():
        users = User.query.order_by(User.id.asc()).all()
        return render_template("admin.html", users=users)

    def form_grant_values():
        try:
            days = int(request.form.get("days", ""))
            hours = int(request.form.get("hours", ""))
            minutes = int(request.form.get("minutes", ""))
            seconds = int(request.form.get("seconds", ""))
            limit = int(request.form.get("analysis_limit", ""))
        except (TypeError, ValueError):
            raise ValueError("Enter whole numbers for the access duration and limit.")

        if (
            days < 0
            or days > 36500
            or not 0 <= hours <= 23
            or not 0 <= minutes <= 59
            or not 0 <= seconds <= 59
        ):
            raise ValueError("Enter a valid duration (up to 100 years).")
        duration = timedelta(
            days=days, hours=hours, minutes=minutes, seconds=seconds
        )
        if duration.total_seconds() <= 0:
            raise ValueError("Access duration must be at least one second.")
        if not 1 <= limit <= 1_000_000:
            raise ValueError("Analysis limit must be between 1 and 1,000,000.")
        return utcnow() + duration, limit

    @app.route("/admin/users/<int:user_id>/approve", methods=["POST"])
    @admin_required
    def approve_user(user_id):
        user = db.get_or_404(User, user_id)
        if user.role == "owner":
            abort(400)
        if user.status not in {"pending", "rejected", "suspended"}:
            flash("This account is already active.", "info")
            return redirect(url_for("admin_dashboard"))
        try:
            user.expires_at, user.analysis_limit = form_grant_values()
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("admin_dashboard"))

        user.status = "approved"
        db.session.commit()
        flash(f"{user.username} can now use the analyzer.", "success")
        return redirect(url_for("admin_dashboard"))

    @app.route("/admin/users/<int:user_id>/limits", methods=["POST"])
    @admin_required
    def update_user_limits(user_id):
        user = db.get_or_404(User, user_id)
        if user.role == "owner":
            abort(400)
        try:
            user.expires_at, user.analysis_limit = form_grant_values()
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("admin_dashboard"))

        db.session.commit()
        flash(f"Access limits updated for {user.username}.", "success")
        return redirect(url_for("admin_dashboard"))

    @app.route("/admin/users/<int:user_id>/status", methods=["POST"])
    @admin_required
    def update_user_status(user_id):
        user = db.get_or_404(User, user_id)
        status = request.form.get("status")
        if user.role == "owner" or status not in {
            "approved",
            "rejected",
            "suspended",
        }:
            abort(400)
        if status == "approved" and (
            user.expires_at is None or user.analysis_limit is None
        ):
            flash("Assign an expiry and analysis limit before activating this user.", "error")
            return redirect(url_for("admin_dashboard"))
        user.status = status
        db.session.commit()
        flash(f"{user.username}'s status is now {status}.", "success")
        return redirect(url_for("admin_dashboard"))

    @app.errorhandler(403)
    def forbidden(_error):
        return render_template(
            "error.html",
            title="Access restricted",
            message="You don't have permission to view this page.",
        ), 403

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return render_template(
            "error.html",
            title="File too large",
            message="Choose an audio file smaller than 100 MB.",
        ), 413

    return app


def run_analysis_job(app, job_id):
    with ANALYSIS_SLOTS:
        with app.app_context():
            job = db.session.get(AnalysisJob, job_id)
            if job is None:
                logger.error("Analysis job %s no longer exists", job_id)
                return

            job.status = "running"
            db.session.commit()
            user_id = job.user_id
            try:
                result = analyze_audio(Path(job.audio_path))
                result["filename"] = job.filename
                if "path" in result:
                    result["path"] = str(result["path"])
                job.result_json = json.dumps(result)
                job.status = "completed"
                db.session.commit()
            except Exception as error:
                db.session.rollback()
                logger.exception("Audio analysis failed for user %s", user_id)
                job = db.session.get(AnalysisJob, job_id)
                if job is None:
                    logger.error("Analysis job %s disappeared after failure", job_id)
                    return
                job.status = "failed"
                job.error_message = analysis_error_message(error)
                db.session.commit()


def analysis_error_message(error):
    if isinstance(error, ModuleNotFoundError):
        if error.name == "madmom":
            return (
                "The analyzer dependency 'madmom' is missing. Run "
                "'pip install git+https://github.com/CPJKU/madmom' "
                "in this Python environment, then restart the app."
            )
        if error.name == "torch":
            return (
                "PyTorch is missing. Install a compatible PyTorch build "
                "from https://pytorch.org in this Python environment, "
                "then restart the app."
            )
        if error.name == "natten":
            return (
                "A compatible NATTEN build is missing. This analyzer "
                "requires NATTEN < 0.17.5, built for your PyTorch version. "
                "Install that matching build, then restart the app."
            )
    if isinstance(error, ImportError):
        if "natten1dav" in str(error):
            return (
                "The installed NATTEN version is incompatible. This "
                "analyzer requires NATTEN < 0.17.5 and a matching "
                "PyTorch build. Install the compatible version, then "
                "restart the app."
            )
        return (
            "Analysis failed because an analyzer dependency could not be "
            "imported. Check the server log for details."
        )
    return "Analysis failed. Check that the audio is valid and try again."


def analyze_audio(audio_path):
    """Run the analyzer and return JSON-compatible analysis data."""
    import allin1
    from dataclasses import asdict

    result = allin1.analyze(
        audio_path,
        model=os.environ.get("ALLIN1_MODEL", "harmonix-all"),
        multiprocess=False,
    )
    data = asdict(result)
    data["duration"] = max(
        (segment["end"] for segment in data["segments"]), default=0
    )
    return data


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
    )

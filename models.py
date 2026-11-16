from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from extensions import db


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(40), unique=True, nullable=False, index=True)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(16), nullable=False, default="user")
    status = db.Column(db.String(16), nullable=False, default="pending")
    expires_at = db.Column(db.DateTime, nullable=True)
    analysis_limit = db.Column(db.Integer, nullable=True)
    analysis_count = db.Column(db.Integer, nullable=False, default=0)

    @property
    def is_admin(self):
        return self.role in {"owner", "admin"}

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class AnalysisJob(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False, index=True
    )
    filename = db.Column(db.String(255), nullable=False)
    audio_path = db.Column(db.String(1024), nullable=False)
    status = db.Column(db.String(16), nullable=False, default="queued")
    result_json = db.Column(db.Text, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )


class UploadSession(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False, index=True
    )
    filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        index=True,
    )

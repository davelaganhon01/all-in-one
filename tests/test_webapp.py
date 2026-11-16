from io import BytesIO
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from app import ANALYSIS_SLOTS, UPLOAD_CHUNK_BYTES, create_app
from extensions import db
from models import AnalysisJob, UploadSession, User


class ImmediateThread:
    def __init__(self, target, args=(), **_kwargs):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


class DeferredThread:
    def __init__(self, **_kwargs):
        pass

    def start(self):
        pass


@dataclass
class FakeAnalysisResult:
    path: Path
    segments: list


class WebApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(
            {
                "TESTING": True,
                "WTF_CSRF_ENABLED": False,
                "SECRET_KEY": "test-key",
                "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
                "UPLOAD_FOLDER": self.temp_dir.name,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()
            db.engine.dispose()
        self.temp_dir.cleanup()

    def test_analysis_slots_allow_two_jobs_at_once(self):
        self.assertTrue(ANALYSIS_SLOTS.acquire(blocking=False))
        self.assertTrue(ANALYSIS_SLOTS.acquire(blocking=False))
        try:
            self.assertFalse(ANALYSIS_SLOTS.acquire(blocking=False))
        finally:
            ANALYSIS_SLOTS.release()
            ANALYSIS_SLOTS.release()

    def test_chunked_upload_assembles_file_and_queues_analysis(self):
        self.client.post(
            "/register",
            data={
                "username": "owner",
                "email": "owner@example.com",
                "password": "a-very-long-owner-password",
            },
        )
        content = b"a" * (UPLOAD_CHUNK_BYTES + 17)
        response = self.client.post(
            "/uploads",
            json={"filename": "track.wav", "size": len(content)},
        )
        self.assertEqual(response.status_code, 200)
        upload = response.json

        first_chunk = self.client.post(
            f"/uploads/{upload['upload_id']}/chunks/0",
            data=content[:UPLOAD_CHUNK_BYTES],
            content_type="application/octet-stream",
        )
        last_chunk = self.client.post(
            f"/uploads/{upload['upload_id']}/chunks/1",
            data=content[UPLOAD_CHUNK_BYTES:],
            content_type="application/octet-stream",
        )
        self.assertEqual(first_chunk.status_code, 200)
        self.assertEqual(last_chunk.status_code, 200)

        with patch("app.threading.Thread", DeferredThread):
            completed = self.client.post(
                f"/uploads/{upload['upload_id']}/complete"
            )
        self.assertEqual(completed.status_code, 200)
        with self.app.app_context():
            job = AnalysisJob.query.one()
            self.assertEqual(Path(job.audio_path).read_bytes(), content)
            self.assertEqual(UploadSession.query.count(), 0)

    def test_chunked_upload_cannot_be_completed_until_all_chunks_arrive(self):
        self.client.post(
            "/register",
            data={
                "username": "owner",
                "email": "owner@example.com",
                "password": "a-very-long-owner-password",
            },
        )
        response = self.client.post(
            "/uploads",
            json={"filename": "track.wav", "size": UPLOAD_CHUNK_BYTES + 1},
        )
        upload_id = response.json["upload_id"]
        response = self.client.post(f"/uploads/{upload_id}/complete")
        self.assertEqual(response.status_code, 400)
        self.assertIn("incomplete", response.json["error"])

    def test_upload_session_can_be_cancelled(self):
        self.client.post(
            "/register",
            data={
                "username": "owner",
                "email": "owner@example.com",
                "password": "a-very-long-owner-password",
            },
        )
        response = self.client.post(
            "/uploads",
            json={"filename": "track.wav", "size": 1},
        )
        upload_id = response.json["upload_id"]
        response = self.client.delete(f"/uploads/{upload_id}")
        self.assertEqual(response.status_code, 204)
        with self.app.app_context():
            self.assertIsNone(db.session.get(UploadSession, upload_id))
            self.assertFalse(
                (Path(self.temp_dir.name) / ".chunks" / upload_id).exists()
            )

    def test_csrf_is_required_for_mutations(self):
        self.app.config["WTF_CSRF_ENABLED"] = True
        response = self.client.post(
            "/register",
            data={
                "username": "listener",
                "email": "listener@example.com",
                "password": "a-very-long-listener-password",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_analyzer_dependencies_have_setup_messages(self):
        self.client.post(
            "/register",
            data={
                "username": "owner",
                "email": "owner@example.com",
                "password": "a-very-long-owner-password",
            },
        )
        setup_messages = {
            "madmom": b"pip install git+https://github.com/CPJKU/madmom",
            "natten": b"0.17.5",
        }
        for dependency, setup_message in setup_messages.items():
            with self.subTest(dependency=dependency):
                with patch(
                    "app.analyze_audio",
                    side_effect=ModuleNotFoundError(
                        f"No module named '{dependency}'", name=dependency
                    ),
                ), patch("app.threading.Thread", ImmediateThread):
                    with self.assertLogs("app", level="ERROR"):
                        response = self.client.post(
                            "/dashboard",
                            data={"audio": (BytesIO(b"audio bytes"), "track.wav")},
                            content_type="multipart/form-data",
                            follow_redirects=True,
                        )
                self.assertIn(dependency.encode(), response.data.lower())
                self.assertIn(b"is missing", response.data)
                self.assertIn(setup_message, response.data)

    def test_analysis_uses_full_ensemble_by_default(self):
        fake_allin1 = ModuleType("allin1")
        fake_allin1.analyze = lambda *args, **kwargs: FakeAnalysisResult(
            path=Path("track.wav"),
            segments=[{"end": 42}],
        )
        with patch.dict(sys.modules, {"allin1": fake_allin1}), patch.dict(
            os.environ, {}, clear=False
        ) as environment:
            environment.pop("ALLIN1_MODEL", None)
            with patch.object(fake_allin1, "analyze", wraps=fake_allin1.analyze) as analyzer:
                from app import analyze_audio

                analyze_audio(Path("track.wav"))

        self.assertEqual(analyzer.call_args.kwargs["model"], "harmonix-all")
        self.assertFalse(analyzer.call_args.kwargs["multiprocess"])

    def test_analysis_model_can_be_overridden(self):
        fake_allin1 = ModuleType("allin1")
        fake_allin1.analyze = lambda *args, **kwargs: FakeAnalysisResult(
            path=Path("track.wav"),
            segments=[],
        )
        with patch.dict(sys.modules, {"allin1": fake_allin1}), patch.dict(
            os.environ, {"ALLIN1_MODEL": "harmonix-fold0"}
        ):
            with patch.object(fake_allin1, "analyze", wraps=fake_allin1.analyze) as analyzer:
                from app import analyze_audio

                analyze_audio(Path("track.wav"))

        self.assertEqual(analyzer.call_args.kwargs["model"], "harmonix-fold0")

    def test_incompatible_natten_version_has_a_version_message(self):
        self.client.post(
            "/register",
            data={
                "username": "owner",
                "email": "owner@example.com",
                "password": "a-very-long-owner-password",
            },
        )
        with patch(
            "app.analyze_audio",
            side_effect=ImportError(
                "cannot import name 'natten1dav' from 'natten.functional'"
            ),
        ), patch("app.threading.Thread", ImmediateThread):
            with self.assertLogs("app", level="ERROR"):
                response = self.client.post(
                    "/dashboard",
                    data={"audio": (BytesIO(b"audio bytes"), "track.wav")},
                    content_type="multipart/form-data",
                    follow_redirects=True,
                )
        self.assertIn(b"NATTEN version is incompatible", response.data)
        self.assertIn(b"0.17.5", response.data)

    def test_analysis_is_queued_and_job_status_is_private(self):
        self.client.post(
            "/register",
            data={
                "username": "owner",
                "email": "owner@example.com",
                "password": "a-very-long-owner-password",
            },
        )
        with patch("app.analyze_audio") as analyzer, patch(
            "app.threading.Thread", DeferredThread
        ):
            response = self.client.post(
                "/dashboard",
                data={"audio": (BytesIO(b"audio bytes"), "track.wav")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 302)
        analyzer.assert_not_called()
        job_url = response.location
        response = self.client.get(f"{job_url}/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "queued")

        self.client.post("/logout")
        self.client.post(
            "/register",
            data={
                "username": "listener",
                "email": "listener@example.com",
                "password": "a-very-long-listener-password",
            },
        )
        self.client.post(
            "/login",
            data={
                "identity": "listener",
                "password": "a-very-long-listener-password",
            },
        )
        response = self.client.get(f"{job_url}/status")
        self.assertEqual(response.status_code, 403)

    def test_owner_approval_and_analysis_limit(self):
        response = self.client.post(
            "/register",
            data={
                "username": "owner",
                "email": "owner@example.com",
                "password": "a-very-long-owner-password",
            },
        )
        self.assertEqual(response.status_code, 302)
        with self.app.app_context():
            owner = User.query.one()
            self.assertEqual(owner.role, "owner")
            self.assertEqual(owner.status, "approved")

        self.client.post("/logout")
        self.client.post(
            "/register",
            data={
                "username": "listener",
                "email": "listener@example.com",
                "password": "a-very-long-listener-password",
            },
        )
        self.client.post(
            "/login",
            data={
                "identity": "listener",
                "password": "a-very-long-listener-password",
            },
        )
        response = self.client.get("/dashboard")
        self.assertIn(b"waiting for owner approval", response.data)
        self.assertNotIn(b"Run analysis", response.data)
        self.client.post("/logout")
        self.client.post(
            "/login",
            data={"identity": "owner", "password": "a-very-long-owner-password"},
        )
        response = self.client.get("/admin/users")
        self.assertEqual(response.status_code, 200)

        with self.app.app_context():
            listener = User.query.filter_by(username="listener").one()
            listener_id = listener.id
        self.client.post(
            f"/admin/users/{listener_id}/approve",
            data={
                "days": 0,
                "hours": 0,
                "minutes": 2,
                "seconds": 0,
                "analysis_limit": 1,
            },
        )
        with self.app.app_context():
            listener = db.session.get(User, listener_id)
            self.assertEqual(listener.status, "approved")
            self.assertEqual(listener.analysis_limit, 1)
            self.assertIsNotNone(listener.expires_at)

        self.client.post("/logout")
        self.client.post(
            "/login",
            data={
                "identity": "listener",
                "password": "a-very-long-listener-password",
            },
        )
        fake_result = {
            "bpm": 120,
            "beats": [0.5, 1.0],
            "downbeats": [0.5],
            "segments": [{"start": 0.0, "end": 1.0, "label": "intro"}],
            "duration": 1.0,
        }
        with patch("app.analyze_audio", return_value=fake_result), patch(
            "app.threading.Thread", ImmediateThread
        ):
            response = self.client.post(
                "/dashboard",
                data={"audio": (BytesIO(b"audio bytes"), "track.wav")},
                content_type="multipart/form-data",
            )
            self.assertEqual(response.status_code, 302)
            job_url = response.location
            response = self.client.get(job_url)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"120", response.data)
            with self.app.app_context():
                job = AnalysisJob.query.one()
                self.assertEqual(job.status, "completed")
            status_response = self.client.get(f"{job_url}/status")
            self.assertEqual(status_response.status_code, 200)
            self.assertEqual(status_response.json["status"], "completed")

            response = self.client.post(
                "/dashboard",
                data={"audio": (BytesIO(b"another track"), "another.wav")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )
            self.assertIn(b"used all of your allowed analyses", response.data)

        with self.app.app_context():
            listener = db.session.get(User, listener_id)
            self.assertEqual(listener.analysis_count, 1)
            listener.expires_at = (
                datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
            )
            db.session.commit()
        response = self.client.get("/dashboard")
        self.assertIn(b"account access has expired", response.data)
        self.assertNotIn(b"Run analysis", response.data)


if __name__ == "__main__":
    unittest.main()

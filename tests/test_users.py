from src.users import LocalUserStore


def test_initial_admin_and_authentication(tmp_path):
    store = LocalUserStore(tmp_path / "users.sqlite3")

    admin = store.create_initial_admin(
        user_id="Lab.Admin",
        display_name="Lab Admin",
        password="correct-horse",
    )

    assert admin.user_id == "lab.admin"
    assert admin.role == "admin"
    assert store.authenticate("lab.admin", "wrong-password") is None
    authenticated = store.authenticate("lab.admin", "correct-horse")
    assert authenticated is not None
    assert authenticated.last_login_at


def test_admin_can_create_and_update_student_account(tmp_path):
    store = LocalUserStore(tmp_path / "users.sqlite3")
    store.create_initial_admin(user_id="admin", display_name="Admin", password="admin-pass")

    student = store.create_user(
        user_id="student.1",
        display_name="Student One",
        password="student-pass",
    )
    updated = store.update_user(
        student.user_id,
        display_name="Student 1",
        role="student",
        active=False,
    )

    assert updated.active is False
    assert store.authenticate("student.1", "student-pass") is None
    assert [user.user_id for user in store.list_users()] == ["admin"]
    assert {user.user_id for user in store.list_users(include_inactive=True)} == {
        "admin",
        "student.1",
    }


def test_last_active_admin_cannot_be_disabled(tmp_path):
    store = LocalUserStore(tmp_path / "users.sqlite3")
    store.create_initial_admin(user_id="admin", display_name="Admin", password="admin-pass")

    try:
        store.update_user("admin", display_name="Admin", role="student", active=True)
    except ValueError as exc:
        assert str(exc) == "At least one active admin is required."
    else:
        raise AssertionError("expected last-admin guard")


def test_query_history_is_isolated_by_user(tmp_path):
    store = LocalUserStore(tmp_path / "users.sqlite3")
    store.create_initial_admin(user_id="admin", display_name="Admin", password="admin-pass")
    store.create_user(
        user_id="student",
        display_name="Student",
        password="student-pass",
    )

    store.record_query(
        user_id="admin",
        question="Admin question",
        answer="Admin answer",
        answer_mode="explanation",
    )
    student_entry = store.record_query(
        user_id="student",
        question="Student question",
        answer="Student answer",
        answer_mode="derivation",
    )

    assert [entry.question for entry in store.list_history("admin")] == ["Admin question"]
    assert store.list_history("student") == [student_entry]
    assert store.clear_history("student") == 1
    assert store.list_history("student") == []
    assert len(store.list_history("admin")) == 1


def test_password_can_be_reset(tmp_path):
    store = LocalUserStore(tmp_path / "users.sqlite3")
    store.create_initial_admin(user_id="admin", display_name="Admin", password="old-password")

    store.set_password("admin", "new-password")

    assert store.authenticate("admin", "old-password") is None
    assert store.authenticate("admin", "new-password") is not None

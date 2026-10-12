"""Consent dialog: all off by default, remote disabled, home folder refused."""

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from soma.consent import ConsentManager, Scope
from soma.ui.desktop.consent_dialog import HOME_REFUSAL, REMOTE_NOTE, ConsentDialog


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_defaults_all_off(qapp):
    consent = ConsentManager()
    dlg = ConsentDialog(consent, pick_folder=lambda _p: "")
    assert set(dlg.checkboxes) == set(Scope)
    assert not any(b.isChecked() for b in dlg.checkboxes.values())
    assert not dlg.capture_granted
    assert not dlg.folder_btn.isEnabled()


def test_remote_inference_disabled(qapp):
    dlg = ConsentDialog(ConsentManager(), pick_folder=lambda _p: "")
    box = dlg.checkboxes[Scope.REMOTE_INFERENCE]
    assert not box.isEnabled() and REMOTE_NOTE in box.text()
    box.setChecked(True)
    dlg.apply()
    assert not dlg.consent.allowed(Scope.REMOTE_INFERENCE)


def test_saving_without_capture_grants_nothing(qapp):
    consent = ConsentManager()
    dlg = ConsentDialog(consent, pick_folder=lambda _p: "")
    dlg.accept()
    assert not any(consent.allowed(s) for s in Scope)


def test_scopes_are_independent(qapp):
    consent = ConsentManager()
    dlg = ConsentDialog(consent, pick_folder=lambda _p: "")
    dlg.checkboxes[Scope.FEEDBACK_STORAGE].setChecked(True)
    dlg.accept()
    assert consent.allowed(Scope.FEEDBACK_STORAGE)
    assert not consent.allowed(Scope.SCREEN_CAPTURE)
    dlg2 = ConsentDialog(consent, pick_folder=lambda _p: "")
    dlg2.checkboxes[Scope.SCREEN_CAPTURE].setChecked(True)
    dlg2.accept()
    assert consent.allowed(Scope.SCREEN_CAPTURE) and consent.allowed(Scope.FEEDBACK_STORAGE)


def test_home_folder_refused(qapp):
    consent = ConsentManager()
    dlg = ConsentDialog(consent, pick_folder=lambda _p: str(Path.home()))
    dlg.checkboxes[Scope.FOLDER_INDEXING].setChecked(True)
    assert dlg.folder_btn.isEnabled()
    assert dlg.choose_folder() is None
    assert dlg.folder_msg.text() == HOME_REFUSAL
    assert consent.allowed_folders == [] and dlg.chosen_folders == []


def test_folder_added_and_granted_on_save(qapp, tmp_path):
    consent = ConsentManager()
    dlg = ConsentDialog(consent, pick_folder=lambda _p: str(tmp_path))
    dlg.checkboxes[Scope.FOLDER_INDEXING].setChecked(True)
    added = dlg.choose_folder()
    assert added == str(tmp_path.resolve())
    assert dlg.folder_list.count() == 1
    dlg.accept()
    assert consent.path_allowed(tmp_path)
    assert dlg.chosen_folders == [added]


def test_paused_capture_stays_checked(qapp):
    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    consent.pause()
    dlg = ConsentDialog(consent, pick_folder=lambda _p: "")
    assert dlg.checkboxes[Scope.SCREEN_CAPTURE].isChecked()

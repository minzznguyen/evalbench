import os
import pytest
from unittest.mock import patch, MagicMock
from generators.models.gemini_cli import GeminiCliGenerator


@patch('generators.models.gemini_cli.logging')
@patch('generators.models.gemini_cli.subprocess.run')
@patch('generators.models.gemini_cli.os.path.exists')
@patch('generators.models.gemini_cli.shutil.copytree')
@patch('generators.models.gemini_cli.shutil.rmtree')
def test_setup_single_skill_string(
    mock_rmtree, 
    mock_copytree, 
    mock_exists, 
    mock_run,
    mock_logging
):
    """Test that a single skill defined as a string is copied to fake home."""
    print("\n--- STARTING TEST: test_setup_single_skill_string ---")
    
    real_home = "/home/jamesamn"
    real_skill_path = os.path.join(real_home, ".gemini", "skills", "my-single-skill")
    print(f"[Setup] Real home directory set to: {real_home}")
    print(f"[Setup] Expected path of the skill in real home: {real_skill_path}")
    
    # We mock 'os.path.exists' because we don't want the test to actually look 
    # at your real disk. We want to simulate different scenarios.
    def exists_side_effect(path):
        print(f"[Mock os.path.exists] Checking if path exists: {path}")
        if path == real_skill_path:
            print(f"[Mock os.path.exists] Path MATCHES real skill path! Returning True (Simulating it exists).")
            return True
        print(f"[Mock os.path.exists] Path does NOT match. Returning False.")
        return False
    mock_exists.side_effect = exists_side_effect

    # We mock 'subprocess.run' because during setup, the generator tries to
    # run 'gcloud auth print-access-token' to authenticate NPM.
    # We simulate a successful run returning a fake token.
    mock_run.return_value = MagicMock(returncode=0, stdout="fake-token")
    print("[Mock subprocess.run] Configured to simulate successful 'gcloud auth' (returns 'fake-token')")

    # This is the configuration we are passing to the generator.
    # We are telling it to set up a skill named "my-single-skill".
    config = {
        "setup": {
            "skills": ["my-single-skill"]
        }
    }
    print(f"[Config] Generator configuration: {config}")

    print("[Execution] Initializing GeminiCliGenerator (this will trigger _setup and _setup_skills)...")
    # We patch 'os.makedirs' and 'open' to prevent the generator from actually 
    # creating folders and files on your disk during this test.
    with patch('generators.models.gemini_cli.os.makedirs'), \
         patch('generators.models.gemini_cli.open', create=True), \
         patch.dict(os.environ, {"HOME": real_home}):
        
        GeminiCliGenerator(config)
    print("[Execution] GeminiCliGenerator initialization complete.")

    # Now we verify if the generator did what we expected.
    # It should have copied the skill from the 'real home' to the 'fake home'.
    expected_fake_home = os.path.abspath(os.path.join(".venv", "fake_home"))
    expected_fake_skill_path = os.path.join(expected_fake_home, ".gemini", "skills", "my-single-skill")
    
    print(f"[Assertion] Checking if shutil.copytree was called to copy the skill...")
    print(f"[Assertion] Expected Source (Real): {real_skill_path}")
    print(f"[Assertion] Expected Destination (Fake): {expected_fake_skill_path}")

    try:
        mock_copytree.assert_called_once_with(real_skill_path, expected_fake_skill_path)
        print("[Assertion] SUCCESS: shutil.copytree was called with the exact expected paths!")
    except AssertionError as e:
        print("[Assertion] FAILED: shutil.copytree was NOT called as expected!")
        raise e

    print("[Assertion] Checking if correct log message was logged...")
    try:
        mock_logging.info.assert_any_call("Syncing skill: my-single-skill")
        print("[Assertion] SUCCESS: Log message 'Syncing skill: my-single-skill' was found!")
    except AssertionError as e:
        print("[Assertion] FAILED: Expected log message was NOT found!")
        raise e
        
    print("--- END OF TEST: test_setup_single_skill_string ---")


@patch('generators.models.gemini_cli.subprocess.run')
def test_skill_content_preserved(mock_run, tmp_path, monkeypatch):
    """Test that the content of the skill is preserved when copied to fake home."""
    print("\n--- STARTING TEST: test_skill_content_preserved ---")

    # 1. Setup paths in temp directory
    real_home = tmp_path / "real_home"
    real_home.mkdir()

    skill_name = "my-content-skill"
    real_skill_dir = real_home / ".gemini" / "skills" / skill_name
    real_skill_dir.mkdir(parents=True)

    # Create a file inside the skill with some content
    skill_file = real_skill_dir / "secret.txt"
    expected_content = "password is xyz"
    skill_file.write_text(expected_content)

    print(f"[Setup] Created real home at: {real_home}")
    print(f"[Setup] Created skill at: {real_skill_dir}")
    print(f"[Setup] Created skill file with content: {expected_content}")

    # 2. Mock subprocess.run for gcloud auth
    mock_run.return_value = MagicMock(returncode=0, stdout="fake-token")
    print("[Mock subprocess.run] Configured to simulate successful 'gcloud auth'")

    # 3. Configure CWD and HOME env var to use temp directory
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(real_home))
    print(f"[Setup] Changed CWD to: {tmp_path}")
    print(f"[Setup] Set HOME to: {real_home}")

    config = {
        "setup": {
            "skills": [skill_name]
        }
    }

    print("[Execution] Initializing GeminiCliGenerator (real file operations)...")
    # We do NOT patch os.makedirs, open, shutil, or os.path.exists here.
    # They will run for real inside the tmp_path.
    GeminiCliGenerator(config)
    print("[Execution] GeminiCliGenerator initialization complete.")

    # 4. Verify if the file was copied and content preserved
    expected_fake_skill_file = tmp_path / ".venv" / "fake_home" / ".gemini" / "skills" / skill_name / "secret.txt"

    print(f"[Assertion] Checking if copied file exists at: {expected_fake_skill_file}")
    assert expected_fake_skill_file.exists(), f"Copied skill file not found at {expected_fake_skill_file}"
    print("[Assertion] SUCCESS: Copied skill file exists!")

    print("[Assertion] Checking if content is preserved...")
    actual_content = expected_fake_skill_file.read_text()
    assert actual_content == expected_content, f"Content mismatch. Expected: '{expected_content}', Got: '{actual_content}'"
    print(f"[Assertion] SUCCESS: Content matches! Got: '{actual_content}'")

    print("--- END OF TEST: test_skill_content_preserved ---")

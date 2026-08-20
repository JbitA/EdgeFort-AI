import json
from edgeai.cli import main

def test_cli_evaluate_file(tmp_path):
 p=tmp_path/'x.json';assert main(['evaluate','--output',str(p)])==0;d=json.loads(p.read_text());assert 'deployment' in d

def test_cli_showcase_smoke(tmp_path):
    out = tmp_path / "showcase"
    assert main(["showcase", "--dataset", "digits", "--output-dir", str(out), "--benchmark-iterations", "3"]) == 0
    assert (out / "showcase.json").is_file()


def test_cli_dataset_fetch_routes_to_acquirer(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "har"
    monkeypatch.setattr("edgeai.cli.fetch_uci_har", lambda path, force=False: destination)
    assert main(["dataset", "fetch", "uci-har", "--destination", str(destination)]) == 0
    assert str(destination) in capsys.readouterr().out

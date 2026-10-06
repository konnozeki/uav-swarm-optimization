import csv

from prepare_external_datasets import main as prepare_main
from src.scenario_io import read_scenario


def write_points(path, count):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["lat", "lon", "category", "samples"],
        )
        writer.writeheader()
        for i in range(count):
            writer.writerow(
                dict(
                    lat=21.0 + 0.001 * (i % 8),
                    lon=105.0 + 0.001 * (i // 8),
                    category="hospital" if i % 7 == 0 else "shop",
                    samples=10 + i,
                )
            )


def test_prepare_external_dataset_cli(tmp_path, monkeypatch):
    osm = tmp_path / "osm.csv"
    cell = tmp_path / "cell.csv"
    mob = tmp_path / "mobility.csv"
    write_points(osm, 30)
    write_points(cell, 30)
    write_points(mob, 30)
    output = tmp_path / "scenarios"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_external_datasets.py",
            "--osm-poi",
            str(osm),
            "--opencellid",
            str(cell),
            "--mobility-csv",
            str(mob),
            "--cases",
            "2",
            "--targets-per-case",
            "12",
            "--output-dir",
            str(output),
        ],
    )

    prepare_main()

    scenarios = sorted(output.rglob("*.json"))
    assert len(scenarios) == 6
    loaded = read_scenario(scenarios[0])
    assert len(loaded.targets) == 12
    assert loaded.width == 1000.0
    assert loaded.height == 1000.0


def test_prepare_c2a_pose_dataset_cli(tmp_path, monkeypatch):
    labels = tmp_path / "c2a_labels"
    labels.mkdir()
    (labels / "case_a.txt").write_text(
        "\n".join(
            [
                "0 0.10 0.20 0.05 0.07 2",
                "0 0.90 0.80 0.03 0.04 4",
            ]
        ),
        encoding="utf-8",
    )
    (labels / "empty.txt").write_text("", encoding="utf-8")
    output = tmp_path / "scenarios"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_external_datasets.py",
            "--c2a-label-dir",
            str(labels),
            "--c2a-min-targets",
            "2",
            "--output-dir",
            str(output),
        ],
    )

    prepare_main()

    scenarios = sorted(output.rglob("*.json"))
    assert len(scenarios) == 1
    loaded = read_scenario(scenarios[0])
    assert loaded.pattern == "external_c2a_pose"
    assert loaded.targets.tolist() == [[100.0, 200.0], [900.0, 800.0]]
    assert loaded.target_weights.tolist() == [3.0, 1.2]

import pandas as pd

from gem_worldmodel.data.validate import validate_labeled_corpus


def test_validate_reports_correct_counts():
    df = pd.DataFrame(
        {
            "species": ["a", "b", "c", "d"],
            "doubling_time_hours": [2.0, 8.0, 3.0, 12.0],
            "growth_temp_c": [30.0, None, 25.0, 20.0],
            "in_gtdb_tree": [True, True, True, False],
        }
    )
    cfg = {"doubling_time_split_hours": 5.0}
    report = validate_labeled_corpus(df, cfg)

    assert report["n_total_labeled_rows"] == 4
    assert report["n_usable_rows"] == 3  # excludes the row without a GTDB placement
    assert report["n_fast_below_split"] == 2  # 2.0 and 3.0
    assert report["n_slow_above_split"] == 1  # 8.0 (row d is excluded via in_gtdb_tree)
    assert report["n_with_temperature_metadata"] == 2

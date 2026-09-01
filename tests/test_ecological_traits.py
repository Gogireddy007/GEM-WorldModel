import numpy as np
import pandas as pd

from gem_worldmodel.features.ecological_traits import classify_isolation_source, real_trophic_label


def test_classify_isolation_source_copiotroph():
    assert classify_isolation_source("host_animal_endotherm") == 1
    assert classify_isolation_source("wastewater") == 1
    assert classify_isolation_source("food") == 1


def test_classify_isolation_source_oligotroph():
    assert classify_isolation_source("water_marine") == 0
    assert classify_isolation_source("rock_deep") == 0


def test_classify_isolation_source_ambiguous_returns_none():
    assert classify_isolation_source("soil") is None
    assert classify_isolation_source("sediment") is None
    assert classify_isolation_source(np.nan) is None


def test_real_trophic_label_aligns_to_input_index():
    species = pd.Series(
        ["Alpha bacterium", "Beta bacterium", "Gamma bacterium", "Delta bacterium"],
        index=[10, 20, 30, 40],
    )
    madin_traits = pd.DataFrame(
        {
            "species": ["Alpha bacterium", "Beta bacterium", "Gamma bacterium"],
            "isolation_source": ["host_animal_endotherm", "water_marine", "soil"],
        }
    )
    labels = real_trophic_label(species, madin_traits)
    assert list(labels.index) == [10, 20, 30, 40]
    assert labels.loc[10] == 1
    assert labels.loc[20] == 0
    assert pd.isna(labels.loc[30])
    assert pd.isna(labels.loc[40])  # no match in madin_traits at all

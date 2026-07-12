from __future__ import annotations

import numpy as np

from hailmary.actions.speed import realize_speed_variant
from hailmary.templates import TemplateStore

from .test_templates import _template


def test_template_store_round_trip_refreezes_arrays_and_preserves_hashes(tmp_path) -> None:
    template = _template()
    store = TemplateStore((template,))
    modified = realize_speed_variant(
        template.baseline_variant,
        anchor_s_m=80_000.0,
        band="light",
        reduction_kts=10.0,
    )
    cache_key = store.cache_key("speed", template.baseline_variant.variant_id, 80_000.0, "light")
    store.add_variant(modified, cache_key=cache_key)

    path = store.write(tmp_path)
    loaded = TemplateStore.read(path)

    assert loaded.templates[0].template_id == template.template_id
    assert {item.variant_id for item in loaded.variants} == {item.variant_id for item in store.variants}
    reloaded_variant = loaded.variant(modified.variant_id)
    assert reloaded_variant.s_m.dtype == np.float64
    assert reloaded_variant.s_m.flags.c_contiguous
    assert not reloaded_variant.s_m.flags.writeable
    assert loaded.get_or_compile(cache_key, lambda: (_ for _ in ()).throw(AssertionError())) is reloaded_variant

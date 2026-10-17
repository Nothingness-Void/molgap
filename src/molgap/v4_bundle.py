"""Historical packaging names only; new packages use source_bundle."""
from .source_bundle import TEXT_SUFFIXES, _payload, build_source_bundle

build_v4_source_bundle = build_source_bundle

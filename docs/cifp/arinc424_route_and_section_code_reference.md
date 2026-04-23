# ARINC-424 Route Type and Section Code Reference

This note records the `route_type` and `section_code_2` values used by the CIFP parser in this repository.

The extractor stores the cleaned, decoded ARINC-424 field values:

- `route_type` comes from `Route Type`
- `section_code_2` comes from `Section Code (2)`

Source in code:

- [`src/scenario/cifp_parser/airport_related_extractor.py`](../src/scenario/cifp_parser/airport_related_extractor.py)
- `arinc424` decoder table installed in the environment

## Route Type

The meaning of `route_type` depends on the record section:

- `PD` and `HD` use the `SID` table
- `PE` and `HE` use the `STAR` table
- `PF` and `HF` use the `APPR` table
- `ER` and `ET` use the enroute route tables

### SID / departure procedures

Used for `PD` and `HD`.

| Value | Meaning |
| --- | --- |
| `0` | Engine Out SID |
| `1` | SID Runway Transition |
| `2` | SID or SID Common Route |
| `3` | SID Enroute Transition |
| `4` | RNAV SID Runway Transition |
| `5` | RNAV SID or SID Common Route |
| `6` | RNAV SID Enroute Transition |
| `F` | FMS SID Runway Transition |
| `M` | FMS SID or SID Common Route |
| `S` | FMS SID Enroute Transition |
| `R` | RNP SID Runway Transition |
| `N` | RNP SID or SID Common Route |
| `P` | RNP SID Enroute Transition |
| `T` | Vector SID Runway Transition |
| `V` | Vector SID Enroute Transition |

### STAR / arrival procedures

Used for `PE` and `HE`.

| Value | Meaning |
| --- | --- |
| `1` | STAR Enroute Transition |
| `2` | STAR or STAR Common Route |
| `3` | STAR Runway Transition |
| `4` | RNAV STAR Enroute Transition |
| `5` | RNAV STAR or STAR Common Route |
| `6` | RNAV STAR Runway Transition |
| `7` | Profile Descent Enroute Transition |
| `8` | Profile Descent Common Route |
| `9` | Profile Descent Runway Transition |
| `F` | FMS STAR Enroute Transition |
| `M` | FMS STAR or STAR Common Route |
| `S` | FMS STAR Runway Transition |
| `R` | RNP STAR Enroute Transition |
| `N` | RNP STAR or STAR Common Route |
| `P` | RNP STAR Runway Transition |

### Approach procedures

Used for `PF` and `HF`.

| Value | Meaning |
| --- | --- |
| `A` | Approach Transition |
| `B` | Localizer/Backcourse Approach |
| `D` | VORDME Approach |
| `F` | Flight Management System (FMS) Approach |
| `G` | Instrument Guidance System (IGS) Approach |
| `H` | Area Navigation (RNAV) Approach with Required Navigation Performance (RNP) Approach |
| `I` | Instrument Landing System (ILS) Approach |
| `J` | GNSS Landing System (GLS) Approach |
| `L` | Localizer Only (LOC) Approach |
| `M` | Microwave Landing System (MLS) Approach |
| `N` | Non-Directional Beacon (NDB) Approach |
| `P` | Global Position System (GPS) Approach |
| `Q` | Non-Directional Beacon + DME (NDB+DME) Approach |
| `R` | Area Navigation (RNAV) Approach |
| `S` | VOR Approach using VORDME/VORTAC |
| `T` | TACAN Approach |
| `U` | Simplified Directional Facility (SDF) Approach |
| `V` | VOR Approach |
| `W` | Microwave Landing System (MLS), Type A Approach |
| `X` | Localizer Directional Aid (LDA) Approach |
| `Y` | Microwave Landing System (MLS), Type B and C Approach |
| `Z` | Missed Approach |

### Enroute routes

Used for `ER` and `ET`.

| Value | Meaning |
| --- | --- |
| `A` | Airline Airway (Tailored Data) |
| `C` | Control |
| `D` | Direct Route |
| `H` | Helicopter Airways |
| `O` | Officially Designated Airways |
| `R` | RNAV Airways |
| `S` | Undesignated ATS Route |

For `ET`:

| Value | Meaning |
| --- | --- |
| `C` | North American Routes for North Atlantic Traffic Common Portion |
| `D` | Preferential Routes |
| `J` | Pacific Oceanic Transition Routes (PACOTS) |
| `M` | RNAV Airways |
| `N` | Undesignated ATS Route |

## Section Code

`section_code_2` is the decoded value of `Section Code (2)`, and in this parser it is often the section that owns the referenced fix or runway.

Common values you will see in procedure extraction:

| Value | Meaning |
| --- | --- |
| `PC` | Airport Terminal Waypoint |
| `EA` | Waypoint |
| `D` | VHF Navaid |
| `DB` | NDB Navaid |
| `PG` | Airport Runway |
| `PD` | Airport SID |
| `PE` | Airport STAR |
| `PF` | Airport Approach Procedure |

Notes:

- In the raw decoder, `D` is stored as `D ` with a trailing space; the parser trims whitespace, so the dataframe contains `D`.
- `section_code_2` is not limited to the values above. The underlying ARINC-424 section table includes many more codes, but these are the ones most relevant to the CIFP procedure extraction path in this repository.


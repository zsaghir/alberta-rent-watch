# Earlier Edmonton unemployment: retrieval and compatibility assessment

Status: **blocked on retrieving the official archived export**, checked October 21, 2026. No historical observations have been merged.

## Source candidates and required selection

- Archived table: **14-10-0096-01**, Labour force characteristics by census metropolitan area, annual, inactive.
- Table page: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410009601&request_locale=en
- Bulk export: https://www150.statcan.gc.ca/n1/tbl/csv/14100096-eng.zip
- The indexed official table page reports release date **2021-01-08**, annual frequency, unemployment rate, both sexes, and age 15 years and over. The required geography is Edmonton, Alberta (CMA).
- Current local source: **14-10-0461-01**, released **2026-01-09**, annual unemployment rate, total gender, age 15+, Edmonton CMA, 2011–2025.
- The current export identifies **14-10-0385** as its predecessor; this remains another source to investigate.

## What was attempted

The official table page returned a Statistics Canada website-unavailable message. The bulk ZIP request returned HTTP success but its body was HTML, not a ZIP archive. A second request with a download query also returned HTML. File-format checks prevented treating these responses as data.

Consequently, no valid original export, Edmonton observations, or numerical overlap comparison could be produced. HTML error responses were not saved as raw datasets. Earlier year coverage and the archive's geography metadata have not been verified directly from a retrieved export.

## Compatibility decision

**Not established.** Do not splice the archive into the master dataset.

The current unemployment export states that its series uses 2021 CMA boundaries. The archived series' boundaries, historical revisions, and population definitions must be checked against official metadata. “Both sexes” versus “total gender” also requires documenting the source's definition changes. Agreement in overlapping values alone would not establish matching geographical definitions.

When the export is accessible:

1. Preserve the complete original ZIP and metadata separately; record retrieval date and a checksum.
2. Extract Edmonton CMA unemployment, both sexes, age 15+, annual percent. Retain 2001–2010 and every overlapping year from 2011 onward.
3. Reject duplicate years, suppression/missing observations in the intended coverage, non-finite numbers, and rates outside 0–100.
4. Compare each shared year with the current source. Save year, archived rate, current rate, and current-minus-archived difference in percentage points. Document revisions and geography differences without choosing an arbitrary tolerance to declare compatibility.
5. Only propose a merged experimental series after the definitions and discontinuities have been assessed.

## Prediction-year implications

The population-growth lag is first available in **2003**. If compatible unemployment includes 2002, the combined model could also begin in 2003. If unemployment begins in 2003, it could begin in 2004. These are conditional start dates, not confirmed data coverage.

The current master CSV, economic experiment, and production forecast remain unchanged.

# Non-Life Insurance Actuarial Mathematics: Python Code

Code for the book *Practical Introduction to Non-Life Insurance Actuarial Mathematics: Claims,
Reserves, Pricing and Reinsurance with Python* by F.-X. Cart-Tanneur.

Every figure in the book is produced by this code, and every figure quoted in the text is
checked by the tests in `tests/`.

## Three ways to use the code

1. **Run in your browser, nothing to install.** Open
   **https://fxcarttanneur.github.io/non-life-actuarial-math-code**: a JupyterLab session runs in your browser (JupyterLite, Python compiled to
   WebAssembly). Open any notebook in `notebooks/` and run its cells. The first run of a
   session takes a few seconds to load Python. Your changes stay in your browser only.
2. **Download.** Use *Code > Download ZIP* on this page, or
   [download the ZIP](https://github.com/fxcarttanneur/non-life-actuarial-math-code/archive/refs/heads/main.zip) directly.
3. **Clone.** `git clone https://github.com/fxcarttanneur/non-life-actuarial-math-code.git`

To run locally (Python 3.11 or later):

```bash
python -m pip install -r requirements.txt
python -m pytest tests          # checks every figure quoted in the book
jupyter lab                     # then open notebooks/
```

## Code by chapter

| Chapter | Module | Notebook | Online |
|---|---|---|---|
| **Part I: Getting Started** | | | |
| 1. What Is General Insurance Mathematics? | [`gibasics.py`](src/gibasics.py) | [`ch01_general_insurance.ipynb`](notebooks/ch01_general_insurance.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch01_general_insurance.ipynb) |
| 2. Preparing Insurance Data with Python | [`gidata.py`](src/gidata.py) | [`ch02_preparing_data.ipynb`](notebooks/ch02_preparing_data.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch02_preparing_data.ipynb) |
| **Part II: Claims and Ultimate Losses** | | | |
| 3. Claim Development Data | [`gitriangle.py`](src/gitriangle.py) | [`ch03_claim_development.ipynb`](notebooks/ch03_claim_development.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch03_claim_development.ipynb) |
| 4. The Development Method | [`gidevelop.py`](src/gidevelop.py) | [`ch04_development_method.ipynb`](notebooks/ch04_development_method.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch04_development_method.ipynb) |
| 5. The Expected Method | [`giexpected.py`](src/giexpected.py) | [`ch05_expected_method.ipynb`](notebooks/ch05_expected_method.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch05_expected_method.ipynb) |
| 6. Bornhuetter–Ferguson and Benktander Methods | [`gibf.py`](src/gibf.py) | [`ch06_bf_benktander.ipynb`](notebooks/ch06_bf_benktander.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch06_bf_benktander.ipynb) |
| 7. Cape Cod and Frequency–Severity Methods | [`gicapecod.py`](src/gicapecod.py) | [`ch07_capecod_freqsev.ipynb`](notebooks/ch07_capecod_freqsev.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch07_capecod_freqsev.ipynb) |
| 8. Selecting an Ultimate Claim Estimate | [`giselect.py`](src/giselect.py) | [`ch08_selecting_ultimate.ipynb`](notebooks/ch08_selecting_ultimate.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch08_selecting_ultimate.ipynb) |
| **Part III: Loss Models** | | | |
| 9. Claim Frequency and Severity Distributions | [`gilossmodels.py`](src/gilossmodels.py) | [`ch09_frequency_severity.ipynb`](notebooks/ch09_frequency_severity.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch09_frequency_severity.ipynb) |
| 10. Coverage Modifications | [`gicoverage.py`](src/gicoverage.py) | [`ch10_coverage_modifications.ipynb`](notebooks/ch10_coverage_modifications.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch10_coverage_modifications.ipynb) |
| 11. Aggregate Loss Models | [`giaggregate.py`](src/giaggregate.py) | [`ch11_aggregate_loss.ipynb`](notebooks/ch11_aggregate_loss.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch11_aggregate_loss.ipynb) |
| 12. Safety Loading and Ruin | [`giruin.py`](src/giruin.py) | [`ch12_safety_loading_ruin.ipynb`](notebooks/ch12_safety_loading_ruin.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch12_safety_loading_ruin.ipynb) |
| **Part IV: From Losses to Premiums** | | | |
| 13. The Pure Premium | [`gipricing.py`](src/gipricing.py) | [`ch13_pure_premium.ipynb`](notebooks/ch13_pure_premium.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch13_pure_premium.ipynb) |
| 14. Trending Claims and Premiums | [`gipricing.py`](src/gipricing.py) | [`ch14_trending.ipynb`](notebooks/ch14_trending.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch14_trending.ipynb) |
| 15. From Indicated Loss Cost to Rate | [`gipricing.py`](src/gipricing.py) | [`ch15_indicated_rate.ipynb`](notebooks/ch15_indicated_rate.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch15_indicated_rate.ipynb) |
| 16. Classification Ratemaking with GLMs | [`giglm.py`](src/giglm.py) | [`ch16_glm.ipynb`](notebooks/ch16_glm.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch16_glm.ipynb) |
| 17. Increased Limits and Deductible Pricing | [`giilf.py`](src/giilf.py) | [`ch17_ilf_deductibles.ipynb`](notebooks/ch17_ilf_deductibles.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch17_ilf_deductibles.ipynb) |
| 18. Credibility in Pricing | [`gicred.py`](src/gicred.py) | [`ch18_credibility.ipynb`](notebooks/ch18_credibility.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch18_credibility.ipynb) |
| 19. Experience Rating, Retrospective Rating and Bonus-Malus | [`giexperience.py`](src/giexperience.py) | [`ch19_experience_rating.ipynb`](notebooks/ch19_experience_rating.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch19_experience_rating.ipynb) |
| **Part V: Practical Reserving Topics** | | | |
| 20. Adjusting Claims Data for Changing Conditions | [`giadjust.py`](src/giadjust.py) | [`ch20_changing_conditions.ipynb`](notebooks/ch20_changing_conditions.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch20_changing_conditions.ipynb) |
| 21. Unallocated Loss Adjustment Expenses | [`giulae.py`](src/giulae.py) | [`ch21_ulae.ipynb`](notebooks/ch21_ulae.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch21_ulae.ipynb) |
| 22. Reserve Variability | [`givar.py`](src/givar.py) | [`ch22_reserve_variability.ipynb`](notebooks/ch22_reserve_variability.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch22_reserve_variability.ipynb) |
| 23. Discounting and Risk Margins | [`gidiscount.py`](src/gidiscount.py) | [`ch23_discounting_risk_margins.ipynb`](notebooks/ch23_discounting_risk_margins.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch23_discounting_risk_margins.ipynb) |
| 24. Premium Liabilities and Actual-to-Expected Monitoring | [`gipremliab.py`](src/gipremliab.py) | [`ch24_premium_liabilities.ipynb`](notebooks/ch24_premium_liabilities.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch24_premium_liabilities.ipynb) |
| **Part VI: Reinsurance in Practice** | | | |
| 25. Reinsurance Basics | [`gireins.py`](src/gireins.py) | [`ch25_reinsurance_basics.ipynb`](notebooks/ch25_reinsurance_basics.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch25_reinsurance_basics.ipynb) |
| 26. Pricing an Excess-of-Loss Cover | [`gireins.py`](src/gireins.py) | [`ch26_pricing_xl.ipynb`](notebooks/ch26_pricing_xl.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch26_pricing_xl.ipynb) |
| 27. Reserving for Reinsurance | [`gireins.py`](src/gireins.py) | [`ch27_reserving_reinsurance.ipynb`](notebooks/ch27_reserving_reinsurance.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch27_reserving_reinsurance.ipynb) |
| **Part VII: A Complete Worked Case** | | | |
| 28. A Complete Worked Case: Pricing and Reserving a General Insurance Portfolio | [`gicase.py`](src/gicase.py) | [`ch28_worked_case.ipynb`](notebooks/ch28_worked_case.ipynb) | [Run in browser](https://fxcarttanneur.github.io/non-life-actuarial-math-code/lab/index.html?path=notebooks/ch28_worked_case.ipynb) |

Each notebook contains the code blocks of its chapter, in an order that runs from top to
bottom. Comments after `#` give the values printed in the book.

## Layout

| Folder | Contents |
|---|---|
| `src/` | The Python modules, one or more per chapter |
| `notebooks/` | One notebook per chapter |
| `tests/` | Tests that reproduce the figures quoted in the book |

All examples use a simulated motor portfolio generated by the code itself (`synthetic_history`
in `src/gitriangle.py`), so no data files are needed and every estimate can be compared with
the true value.

## Notes

- The cross-checks against the `chainladder` package (Chapters 3 and 4) are skipped in the
  browser, where that package is not available; install it locally to run them.
- Simulations use fixed seeds; results match the book exactly with the versions in
  `requirements.txt` and within sampling error otherwise.

## License

The code is released under the MIT License (see `LICENSE`). The text of the book is
copyright F.-X. Cart-Tanneur, all rights reserved.

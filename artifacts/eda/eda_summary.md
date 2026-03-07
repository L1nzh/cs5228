# EDA Summary (train)

- n_train: 2666
- churn_rate: 0.1455

## Missing Values

- None

## Numeric Summary

```
                         count        mean        std    min       25%     50%      75%     max
Account length          2666.0  100.620405  39.563974   1.00   73.0000  100.00  127.000  243.00
Number vmail messages   2666.0    8.021755  13.612277   0.00    0.0000    0.00   19.000   50.00
Total day minutes       2666.0  179.481620  54.210350   0.00  143.4000  179.95  215.900  350.80
Total day calls         2666.0  100.310203  19.988162   0.00   87.0000  101.00  114.000  160.00
Total day charge        2666.0   30.512404   9.215733   0.00   24.3800   30.59   36.700   59.64
Total eve minutes       2666.0  200.386159  50.951515   0.00  165.3000  200.90  235.100  363.70
Total eve calls         2666.0  100.023631  20.161445   0.00   87.0000  100.00  114.000  170.00
Total eve charge        2666.0   17.033072   4.330864   0.00   14.0500   17.08   19.980   30.91
Total night minutes     2666.0  201.168942  50.780323  43.70  166.9250  201.15  236.475  395.00
Total night calls       2666.0  100.106152  19.418459  33.00   87.0000  100.00  113.000  166.00
Total night charge      2666.0    9.052689   2.285120   1.97    7.5125    9.05   10.640   17.77
Total intl minutes      2666.0   10.237022   2.788349   0.00    8.5000   10.20   12.100   20.00
Total intl calls        2666.0    4.467367   2.456195   0.00    3.0000    4.00    6.000   20.00
Total intl charge       2666.0    2.764490   0.752812   0.00    2.3000    2.75    3.270    5.40
Customer service calls  2666.0    1.562641   1.311236   0.00    1.0000    1.00    2.000    9.00
```

## High Correlation Pairs (|corr| >= 0.95)

- Total day minutes vs Total day charge: 1.0000
- Total eve minutes vs Total eve charge: 1.0000
- Total night minutes vs Total night charge: 1.0000
- Total intl minutes vs Total intl charge: 1.0000

## Top Features (Mutual Information)

- num__Total day charge: 0.044622
- num__Total day minutes: 0.042982
- cat__International plan_Yes: 0.035849
- num__Customer service calls: 0.033697
- cat__International plan_No: 0.031587
- num__Number vmail messages: 0.017774
- cat__State_CA: 0.014736
- cat__State_IN: 0.012365
- cat__State_VT: 0.011514
- cat__State_SD: 0.010703
- cat__State_CO: 0.008812
- cat__State_KS: 0.008738
- cat__Voice mail plan_Yes: 0.008285
- cat__State_OK: 0.008166
- cat__State_AK: 0.007179
- cat__State_OR: 0.006807
- cat__State_TX: 0.005607
- cat__State_AL: 0.005476
- cat__State_MN: 0.005361
- cat__Voice mail plan_No: 0.004985

## Plots

- cat_area_code_churn_rate.png
- cat_area_code_freq.png
- cat_international_plan_churn_rate.png
- cat_international_plan_freq.png
- cat_state_churn_rate.png
- cat_state_freq.png
- cat_voice_mail_plan_churn_rate.png
- cat_voice_mail_plan_freq.png
- correlation_heatmap.png
- num_account_length.png
- num_customer_service_calls.png
- num_number_vmail_messages.png
- num_total_day_calls.png
- num_total_day_charge.png
- num_total_day_minutes.png
- num_total_eve_calls.png
- num_total_eve_charge.png
- num_total_eve_minutes.png
- num_total_intl_calls.png
- num_total_intl_charge.png
- num_total_intl_minutes.png
- num_total_night_calls.png
- num_total_night_charge.png
- num_total_night_minutes.png

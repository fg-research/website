.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: An overview of the FRED-MD database
   :keywords: Time Series, Macroeconomics

######################################################################################
An overview of the FRED-MD database
######################################################################################

.. raw:: html

    <p>
    <a href="https://research.stlouisfed.org/econ/mccracken/fred-databases/" target="_blank">FRED-MD</a>
    is an open-source dataset of monthly U.S. macroeconomic indicators maintained by the Federal
    Reserve Bank of St. Louis. The FRED-MD dataset was introduced to provide a common benchmark
    for comparing model performance and to facilitate the reproducibility of research results
    <a href="#references">[1]</a>. The FRED-MD dataset includes eight different categories
    of macroeconomic indicators (see the <a href="#appendix">Appendix</a> for the full list):
    </p>

#. Output and Income
#. Labor Market
#. Consumption and Orders
#. Orders and Inventories
#. Money and Credit
#. Interest Rates and Exchange Rates
#. Prices
#. Stock Market

.. raw:: html

    <p>
    The time series included in the FRED-MD dataset are sourced from the <a href="https://fred.stlouisfed.org/"
    target="_blank">Federal Reserve Economic Data (FRED) database</a>, which is St. Louis Fed’s main, publicly
    available, economic database. The FRED-MD dataset applies different adjustments to the raw data sourced
    from FRED, such as seasonal adjustments, inflation adjustments and backfilling of missing values.
    <p>

The FRED-MD dataset also takes into account data changes and revisions.
For instance, in the main FRED database the same indicator can be released with different
names and, potentially, be reported in different units, over different time periods.
In the FRED-MD dataset each indicator is instead always represented by a single
time series with a unique name and is always reported in the same units.

The FRED-MD dataset was released for the first time in 01-2015.
At the time of its first release, the FRED-MD dataset contained 134 time series.
As of 12-2023, the FRED-MD dataset contains 127 time series.
118 time series are included in all monthly releases from 01-2015 to 12-2023.
The first date included in the FRED-MD dataset is 01-1959, even though a few time series start several years later.

The FRED-MD dataset is updated on a monthly basis. Each monthly release is referred to as a *vintage*.
A different CSV file is released for each month. The CSV files can be downloaded from the URL below,
where ``{year}`` and ``{month}`` are the year and month of the release.

.. code::

    "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{year}-{month}.csv"

Each CSV file contains the data from 01-1959 up to the previous month end.
For instance, the ``01-2015.csv`` file contains the data from 01-1959 to 12-2014,
the ``02-2015.csv`` file contains the data from 01-1959 to 01-2015, and so on.

.. note::

    The datasets released on a monthly basis since 01-2015 are referred to as *real-time vintages*.
    The authors have also made available the datasets from 08-1999 to 12-2014, which are referred to as *historical vintages*.
    The historical vintages can be downloaded from this `link <https://s3.amazonaws.com/files.research.stlouisfed.org/fred-md/Historical_FRED-MD.zip>`__.

The first row of each CSV file includes the codes of the suggested transformations
to be applied to the time series in order to make them stationary
prior to using them in a statistical model. The transformation
codes are defined as follows:

1. no transformation
2. first order difference
3. second order difference
4. logarithm
5. first order logarithmic difference
6. second order logarithmic difference
7. percentage change

.. raw:: html

    <p>
    The FRED-MD dataset has been used extensively for forecasting US inflation.
    In <a href="#references">[2]</a> it was shown that a random forest model trained on the FRED-MD dataset outperforms several standard inflation forecasting models at different forecasting horizons.
    <a href="#references">[3]</a> expanded the analysis in <a href="#references">[2]</a> to include an LSTM model and found that it did not significantly outperform the random forest model.
    <a href="#references">[4]</a> applied different dimension reduction techniques to the FRED-MD dataset in order to forecast US inflation and found that autoencoders provide the best performance.
    In <a href="#references">[5]</a> it was shown that machine learning models trained on the FRED-MD dataset outperform the standard linear regression model in all considered forecasting periods.
    </p>

******************************************
Code
******************************************
In this section, we provide the Python code for downloading and processing the FRED-MD dataset.
We start by importing the dependencies.

.. code:: python

    import os
    import pandas as pd
    import numpy as np

After that we define a function for transforming the time series based on their assigned transformation code.

.. code:: python

    def transform_series(x, tcode):
        '''
        Transform the time series.

        Parameters:
        ______________________________
        x: pandas.Series
            Time series.

        tcode: int.
            Transformation code.
        '''

        if tcode == 1:
            return x
        elif tcode == 2:
            return x.diff()
        elif tcode == 3:
            return x.diff().diff()
        elif tcode == 4:
            return np.log(x)
        elif tcode == 5:
            return np.log(x).diff()
        elif tcode == 6:
            return np.log(x).diff().diff()
        elif tcode == 7:
            return x.pct_change()
        else:
            raise ValueError(f"unknown `tcode` {tcode}")

We can now define a function for downloading and, optionally, transforming the time series.

.. code:: python

    def get_data(year, month, transform=True):
        '''
        Download and (optionally) transform the time series.

        Parameters:
        ______________________________
        year: int
            The year of the dataset vintage.

        month: int.
            The month of the dataset vintage.

        transform: bool.
            Whether the time series should be transformed or not.
        '''

        # get the dataset URL
        file = f"https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{year}-{format(month, '02d')}.csv"

        # get the time series
        data = pd.read_csv(file, skiprows=[1], index_col=0)
        data.columns = [c.upper() for c in data.columns]

        # process the dates
        data = data.loc[pd.notna(data.index), :]
        data.index = pd.date_range(start="1959-01-01", freq="MS", periods=len(data))

        if transform:

            # get the transformation codes
            tcodes = pd.read_csv(file, nrows=1, index_col=0)
            tcodes.columns = [c.upper() for c in tcodes.columns]

            # transform the time series
            data = data.apply(lambda x: transform_series(x, tcodes[x.name].item()))

        return data

We can then use the above function for downloading the 12-2023 dataset vintage as follows:

.. code:: python

    dataset = get_data(year=2023, month=12, transform=False)




.. tip::

    A Python notebook with additional functions for working with the FRED-MD dataset is available in our
    `GitHub repository <https://github.com/fg-research/blog/blob/master/fred-md-overview/fred_md_overview.ipynb>`__.

******************************************
References
******************************************

[1] McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574-589. `doi: 10.1080/07350015.2015.1086655 <https://doi.org/10.1080/07350015.2015.1086655>`__.

[2] Medeiros, M. C., Vasconcelos, G. F., Veiga, Á., & Zilberman, E. (2021). Forecasting inflation in a data-rich environment: the benefits of machine learning methods. *Journal of Business & Economic Statistics*, 39(1), 98-119. `doi: 10.1080/07350015.2019.1637745 <https://doi.org/10.1080/07350015.2019.1637745>`__.

[3] Paranhos, L. (2023). Predicting Inflation with Recurrent Neural Networks. *Working Paper*.

[4] Hauzenberger, N., Huber, F., & Klieber, K. (2023). Real-time inflation forecasting using non-linear dimension reduction techniques. *International Journal of Forecasting*, 39(2), 901-921. `doi: 10.1016/j.ijforecast.2022.03.002 <https://doi.org/10.1016/j.ijforecast.2022.03.002>`__.

[5] Malladi, R. K. (2023). Benchmark Analysis of Machine Learning Methods to Forecast the US Annual Inflation Rate During a High-Decile Inflation Period. *Computational Economics*, 1-41. `doi: 10.1007/s10614-023-10436-w <https://doi.org/10.1007/s10614-023-10436-w>`__.

******************************************
Appendix
******************************************

1. Output and Income
============================================================

==================== ============================================================
Name                 Description
==================== ============================================================
CUMFNS               Capacity Utilization: Manufacturing
INDPRO               IP: Index
IPBUSEQ              IP: Business Equipment
IPCONGD              IP: Consumer Goods
IPDCONGD             IP: Durable Consumer Goods
IPDMAT               IP: Durable Materials
IPFINAL              IP: Final Products (Market Group)
IPFPNSS              IP: Final Products and Nonindustrial Supplies
IPFUELS              IP: Fuels
IPMANSICS            IP: Manufacturing (SIC)
IPMAT                IP: Materials
IPNCONGD             IP: Nondurable Consumer Goods
IPNMAT               IP: Nondurable Materials
IPB51222S            IP: Residential Utilities
RPI                  Real Personal Income
W875RX1              Real personal income ex transfer receipts
==================== ============================================================

*Output and Income (group 1) FRED-MD time series as of 12-2023.*

2. Labor Market
============================================================

==================== ============================================================
Name                 Description
==================== ============================================================
USCONS               All Employees: Construction
DMANEMP              All Employees: Durable goods
USFIRE               All Employees: Financial Activities
USGOOD               All Employees: Goods-Producing Industries
USGOVT               All Employees: Government
MANEMP               All Employees: Manufacturing
CES1021000001        All Employees: Mining and Logging:  Mining
NDMANEMP             All Employees: Nondurable goods
USTRADE              All Employees: Retail Trade
SRVPRD               All Employees: Service-Providing Industries
PAYEMS               All Employees: Total nonfarm
USTPU                All Employees: Trade, Transportation & Utilities
USWTRADE             All Employees: Wholesale Trade
UEMPMEAN             Average Duration of Unemployment (Weeks)
CES2000000008        Avg Hourly Earnings: Construction
CES0600000008        Avg Hourly Earnings: Goods-Producing
CES3000000008        Avg Hourly Earnings: Manufacturing
CES0600000007        Avg Weekly Hours: Goods-Producing
AWHMAN               Avg Weekly Hours: Manufacturing
AWOTMAN              Avg Weekly Overtime Hours: Manufacturing
CE16OV               Civilian Employment
CLF16OV              Civilian Labor Force
UNRATE               Civilian Unemployment Rate
UEMP15OV             Civilians Unemployed - 15 Weeks & Over
UEMPLT5              Civilians Unemployed - Less Than 5 Weeks
UEMP15T26            Civilians Unemployed for 15-26 Weeks
UEMP27OV             Civilians Unemployed for 27 Weeks and Over
UEMP5TO14            Civilians Unemployed for 5-14 Weeks
HWI                  Help-Wanted Index for United States
CLAIMSX              Initial Claims
HWIURATIO            Ratio of Help Wanted/No. Unemployed
==================== ============================================================

*Labor Market (group 2) FRED-MD time series as of 12-2023.*

3. Consumption and Orders
============================================================

==================== ============================================================
Name                 Description
==================== ============================================================
HOUSTMW              Housing Starts, Midwest
HOUSTNE              Housing Starts, Northeast
HOUSTS               Housing Starts, South
HOUSTW               Housing Starts, West
HOUST                Housing Starts: Total New Privately Owned
PERMIT               New Private Housing Permits (SAAR)
PERMITMW             New Private Housing Permits, Midwest (SAAR)
PERMITNE             New Private Housing Permits, Northeast (SAAR)
PERMITS              New Private Housing Permits, South (SAAR)
PERMITW              New Private Housing Permits, West (SAAR)
==================== ============================================================

*Consumption and Orders (group 3) FRED-MD time series as of 12-2023.*

4. Orders and Inventories
============================================================

==================== ============================================================
Name                 Description
==================== ============================================================
UMCSENTX             Consumer Sentiment Index
ACOGNO               New Orders for Consumer Goods
AMDMNOX              New Orders for Durable Goods
ANDENOX              New Orders for Nondefense Capital Goods
CMRMTSPLX            Real Manu. and Trade Industries Sales
DPCERA3M086SBEA      Real personal consumption expenditures
RETAILX              Retail and Food Services Sales
BUSINVX              Total Business Inventories
ISRATIOX             Total Business: Inventories to Sales Ratio
AMDMUOX              Unfilled Orders for Durable Goods
==================== ============================================================

*Orders and Inventories (group 4) FRED-MD time series as of 12-2023.*

5. Money and Credit
============================================================

==================== ============================================================
Name                 Description
==================== ============================================================
BUSLOANS             Commercial and Industrial Loans
DTCOLNVHFNM          Consumer Motor Vehicle Loans Outstanding
M1SL                 M1 Money Stock
M2SL                 M2 Money Stock
BOGMBASE             Monetary Base
CONSPI               Nonrevolving consumer credit to Personal Income
REALLN               Real Estate Loans at All Commercial Banks
M2REAL               Real M2 Money Stock
NONBORRES            Reserves Of Depository Institutions
INVEST               Securities in Bank Credit at All Commercial Banks
DTCTHFNM             Total Consumer Loans and Leases Outstanding
NONREVSL             Total Nonrevolving Credit
TOTRESNS             Total Reserves of Depository Institutions
==================== ============================================================

*Money and Credit (group 5) FRED-MD time series as of 12-2023.*

6. Interest Rates and Exchange Rates
============================================================

==================== ============================================================
Name                 Description
==================== ============================================================
T1YFFM               1-Year Treasury C Minus FEDFUNDS
GS1                  1-Year Treasury Rate
T10YFFM              10-Year Treasury C Minus FEDFUNDS
GS10                 10-Year Treasury Rate
CP3MX                3-Month AA Financial Commercial Paper Rate
COMPAPFFX            3-Month Commercial Paper Minus FEDFUNDS
TB3MS                3-Month Treasury Bill
TB3SMFFM             3-Month Treasury C Minus FEDFUNDS
T5YFFM               5-Year Treasury C Minus FEDFUNDS
GS5                  5-Year Treasury Rate
TB6MS                6-Month Treasury Bill
TB6SMFFM             6-Month Treasury C Minus FEDFUNDS
EXCAUSX              Canada / U.S. Foreign Exchange Rate
FEDFUNDS             Effective Federal Funds Rate
EXJPUSX              Japan / U.S. Foreign Exchange Rate
BAAFFM               Moody's Baa Corporate Bond Minus FEDFUNDS
AAAFFM               Moodys Aaa Corporate Bond Minus FEDFUNDS
AAA                  Moodys Seasoned Aaa Corporate Bond Yield
BAA                  Moodys Seasoned Baa Corporate Bond Yield
EXSZUSX              Switzerland / U.S. Foreign Exchange Rate
TWEXAFEGSMTHX        Trade Weighted U.S. Dollar Index
EXUSUKX              U.S. / U.K. Foreign Exchange Rate
==================== ============================================================

*Interest Rates and Exchange Rates (group 6) FRED-MD time series as of 12-2023.*

7. Prices
============================================================

==================== ============================================================
Name                 Description
==================== ============================================================
CPIAUCSL             CPI: All Items
CPIULFSL             CPI: All Items Less Food
CUSR0000SA0L5        CPI: All items less medical care
CUSR0000SA0L2        CPI: All items less shelter
CPIAPPSL             CPI: Apparel
CUSR0000SAC          CPI: Commodities
CUSR0000SAD          CPI: Durables
CPIMEDSL             CPI: Medical Care
CUSR0000SAS          CPI: Services
CPITRNSL             CPI: Transportation
OILPRICEX            Crude Oil, spliced WTI and Cushing
WPSID62              PPI: Crude Materials
WPSFD49502           PPI: Finished Consumer Goods
WPSFD49207           PPI: Finished Goods
WPSID61              PPI: Intermediate Materials
PPICMM               PPI: Metals and metal products
DDURRG3M086SBEA      Personal Cons. Exp: Durable goods
DNDGRG3M086SBEA      Personal Cons. Exp: Nondurable goods
DSERRG3M086SBEA      Personal Cons. Exp: Services
PCEPI                Personal Cons. Expend.: Chain Index
==================== ============================================================

*Prices (group 7) FRED-MD time series as of 12-2023.*

8. Stock Market
============================================================

==================== ============================================================
Name                 Description
==================== ============================================================
S&P 500              S&Ps Common Stock Price Index: Composite
S&P: INDUST          S&Ps Common Stock Price Index: Industrials
S&P DIV YIELD        S&Ps Composite Common Stock: Dividend Yield
S&P PE RATIO         S&Ps Composite Common Stock: Price-Earnings Ratio
VIXCLSX              VIX
==================== ============================================================

*Stock Market (group 8) FRED-MD time series as of 12-2023.*


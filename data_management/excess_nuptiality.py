"""
This module provides functions to compute excessive nuptiality using approach from (Mediazona, 2022)
"""


import pandas as pd
from sklearn.linear_model import LinearRegression

AGEGROUPS_15_59 = [
    '15-19 лет', '20-24 лет', '25-29 лет',
    '30-34 лет', '35-39 лет', '40-44 лет',
    '45-49 лет', '50-54 лет', '55-59 лет'
]

AGEGROUPS_15_49 = AGEGROUPS_15_59[:7]

EXCLUDE_REGIONS = [
    'Российская Федерация',
    'Архангельская область',
    'Тюменская область'
]


def _male_excess(region, year, df, horizon=5):
    """Compute age-specific excess male mortality vs expected male/female ratio."""
    LinReg = LinearRegression()
    ex_male = []

    for age in AGEGROUPS_15_59:
        male = df[
            (df.Region == region) &
            (df.Age == age) &
            (df.Gender == 'm') &
            (df.Year <= min(year, 2019))
        ][['Year', 'Deaths']].values

        female = df[
            (df.Region == region) &
            (df.Age == age) &
            (df.Gender == 'f') &
            (df.Year <= min(year, 2019))
        ][['Year', 'Deaths']].values

        male_deaths = male[-(horizon + 1):-1, 1]
        female_deaths = female[-(horizon + 1):-1, 1]
        years = male[-(horizon + 1):-1, 0].reshape(-1, 1)

        valid = female_deaths > 0
        male_deaths = male_deaths[valid]
        female_deaths = female_deaths[valid]
        years = years[valid]

        if len(years) == 0:
            ex_male.append(0.0)
            continue

        ratios = male_deaths / female_deaths
        LinReg.fit(years, ratios.reshape(-1, 1))
        ratio_hat = LinReg.predict([[year]])[0][0]
        male_excess = male[-1, 1] - ratio_hat * female[-1, 1]
        ex_male.append(male_excess)

    return ex_male


def calculate_excess_mortality(df):
    """Calculate excess male mortality per region and year."""
    results = []

    for region in df.Region.unique():
        for year in range(2022, 2024):
            ex = _male_excess(region, year, df)
            results.append({
                'Region': region,
                'Year': year,
                'ExcessByAge': ex,
                'TotalExcess_15_49': sum(ex[:7]),
                'TotalExcess_15_59': sum(ex)
            })

    return pd.DataFrame(results)


def load_and_process_population(path):
    """Load and reshape population data from Excel."""
    pop = pd.read_excel(path)
    pop.columns.values[0:2] = ['age', 'Region']
    pop['Region'] = pop['Region'].str.strip()

    pop_long = pop.melt(
        id_vars=['age', 'Region'],
        var_name='year',
        value_name='male_population'
    )

    pop_long['year'] = pd.to_numeric(pop_long['year'], errors='coerce')
    pop_long = pop_long.dropna(subset=['male_population'])
    return pop_long


def _aggregate_population(pop_long):
    """Aggregate total male populations by age groups."""
    pop_1559 = pop_long[pop_long['age'].isin(AGEGROUPS_15_59)]
    pop_1549 = pop_long[pop_long['age'].isin(AGEGROUPS_15_49)]

    pop_1559 = pop_1559.groupby(['Region', 'year'], as_index=False)['male_population'].sum()
    pop_1549 = pop_1549.groupby(['Region', 'year'], as_index=False)['male_population'].sum()

    pop = pop_1559.merge(pop_1549, on=['Region', 'year'], suffixes=['_1559', '_1549'])
    pop.columns = ['Region', 'Year', 'male_pop1559', 'male_pop1549']
    return pop


def _normalize_region_names(df, mapping):
    """Standardize region names based on mapping."""
    df['Region'] = df['Region'].replace(mapping)
    return df


def compute_normalized_metrics(excess_df, pop_df):
    """Merge and normalize excess mortality by population."""
    df = pop_df.merge(excess_df, on=['Region', 'Year'], how='left')
    df = df.dropna()
    df['ex_per100k_1549'] = df['TotalExcess_15_49'] / df['male_pop1549'] * 100_000
    df['ex_per100k_1559'] = df['TotalExcess_15_59'] / df['male_pop1559'] * 100_000
    return df


def extend_pipeline_with_marriages(df, mapping):
    """Add excess marriages per 100k to mortality pipeline."""
    nup = pd.read_excel("../data/data.xls", header=[0, 1])
    nup = nup.drop(columns=nup.columns[0])
    names = pd.read_excel("../data/data.xls").iloc[1:, 0]
    nup.index = names
    nup.index.name = "Region"
    nup.columns = pd.MultiIndex.from_tuples(nup.columns)

    nup_long = nup.stack(level=[0, 1]).reset_index()
    nup_long.columns = ['Region', 'Year', 'Month', 'Marriages']

    month_map = {
        'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
        'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12
    }
    nup_long['Month'] = nup_long['Month'].map(month_map)
    nup_long['Year'] = nup_long['Year'].astype(int)

    baseline = nup_long[
        nup_long['Year'].between(2015, 2019) & nup_long['Month'].between(9, 11)
    ]
    expected = (
        baseline.groupby(['Region', 'Month'])['Marriages']
        .mean()
        .reset_index()
        .rename(columns={'Marriages': 'Expected'})
    )

    expected = expected.merge(
        nup_long[nup_long['Year'] == 2022],
        on=['Region', 'Month'],
        how='left'
    )
    expected['Excess'] = expected['Marriages'] - expected['Expected']

    mar_ex = expected.groupby('Region')['Excess'].sum().reset_index()
    mar_ex.columns = ['Region', 'Excessive Marriages']

    mar_ex['Region'] = mar_ex['Region'].str.strip().replace(mapping)
    df = df.merge(mar_ex, on='Region', how='left')
    df['ex_nup_per100k'] = df['Excessive Marriages'] / df['male_pop1549'] * 100_000
    return df


def main():
    """Run full pipeline to compute excess mortality and marriage rates."""
    death_df = pd.read_csv("../data/deaths-by-age-gender-region-year-1990-2023.csv.gz")
    excess_df = calculate_excess_mortality(death_df)
    excess_df = excess_df[~excess_df['Region'].isin(EXCLUDE_REGIONS)]

    pop_long = load_and_process_population("../data/male_pop.xls")
    pop_df = _aggregate_population(pop_long)

    region_mapping = {
        'Архангельская область (кроме Ненецкого автономного округа)': 'Архангельская область без АО',
        'Ненецкий автономный округ (Архангельская область)': 'Ненецкий АО',
        'Кемеровская область - Кузбасс': 'Кемеровская область',
        'Город Москва столица Российской Федерации город федерального значения': 'Москва',
        'Город Санкт-Петербург город федерального значения': 'Санкт-Петербург',
        'Город федерального значения Севастополь': 'Севастополь',
        'Еврейская автономная область': 'Еврейская АО',
        'Кабардино-Балкарская Республика': 'Кабардино-Балкария',
        'Карачаево-Черкесская Республика': 'Карачаево-Черкесия',
        'Республика Адыгея (Адыгея)': 'Республика Адыгея',
        'Республика Саха (Якутия)': 'Якутия',
        'Республика Северная Осетия-Алания': 'Северная Осетия',
        'Республика Татарстан (Татарстан)': 'Республика Татарстан',
        'Чувашская Республика - Чувашия': 'Чувашская Республика',
        'Тюменская область (кроме Ханты-Мансийского автономного округа-Югры и Ямало-Ненецкого автономного округа)': 'Тюменская область без АО',
        'Ханты-Мансийский автономный округ - Югра (Тюменская область)': 'Ханты-Мансийский АО',
        'Ямало-Ненецкий автономный округ (Тюменская область)': 'Ямало-Hенецкий АО',
        'Чукотский автономный округ': 'Чукотский АО'
    }

    pop_df = _normalize_region_names(pop_df, region_mapping)
    final_df = compute_normalized_metrics(excess_df, pop_df)

    df_full = extend_pipeline_with_marriages(final_df, region_mapping)

    output_path = "../data/excess_mortality_and_marriages.csv"
    df_full.to_csv(output_path, index=False)
    print(f"Saved final results to {output_path}")


if __name__ == "__main__":
    main()

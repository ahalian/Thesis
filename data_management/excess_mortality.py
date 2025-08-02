"""
This module provides functions to compute excessive male mortality using approach from (Kobak et al., 2025)
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
AGEGROUPS_15_59 = ['15-19 лет', '20-24 лет', '25-29 лет',
                   '30-34 лет', '35-39 лет', '40-44 лет',
                   '45-49 лет', '50-54 лет', '55-59 лет']
AGEGROUPS_15_49 = AGEGROUPS_15_59[:7]
EXCLUDE_REGIONS = ['Российская Федерация', 'Архангельская область', 'Тюменская область']


def _filter_by_demographics(df, region, age, gender, year):
    """Filter dataset by region, age, gender, and year threshold."""
    return df[
        (df.Region == region) &
        (df.Age == age) &
        (df.Gender == gender) &
        (df.Year <= min(year, 2019))
    ][['Year', 'Deaths']].values


def _compute_excess_by_age(region, year, df, horizon=5):
    """Compute excess male deaths per age group based on female trend."""
    linreg = LinearRegression()
    ex_male = []

    for age in AGEGROUPS_15_59:
        male = _filter_by_demographics(df, region, age, 'm', year)
        female = _filter_by_demographics(df, region, age, 'f', year)

        male_deaths = male[-(horizon + 1):-1, 1]
        female_deaths = female[-(horizon + 1):-1, 1]
        years = male[-(horizon + 1):-1, 0].reshape(-1, 1)

        valid = female_deaths > 0
        male_deaths = male_deaths[valid]
        female_deaths = female_deaths[valid]
        years = years[valid]

        if len(male_deaths) < 2:
            ex_male.append(0.0)
            continue

        ratios = male_deaths / female_deaths
        linreg.fit(years, ratios.reshape(-1, 1))
        ratio_hat = linreg.predict([[year]])[0][0]
        male_excess = male[-1, 1] - ratio_hat * female[-1, 1]
        ex_male.append(male_excess)

    return ex_male


def calculate_excess_mortality(df):
    """Calculate excess male mortality for all regions and years."""
    results = []

    for region in df.Region.unique():
        for year in range(2022, 2024):
            ex = _compute_excess_by_age(region, year, df)
            results.append({
                'Region': region,
                'Year': year,
                'ExcessByAge': ex,
                'TotalExcess_15_49': sum(ex[:7]),
                'TotalExcess_15_59': sum(ex)
            })

    return pd.DataFrame(results)


def load_and_process_population(filepath):
    """Load and reshape male population data from Excel."""
    male_pop = pd.read_excel(filepath)
    male_pop.columns.values[0:2] = ['age', 'Region']
    male_pop['Region'] = male_pop['Region'].str.strip()

    male_pop_long = male_pop.melt(
        id_vars=['age', 'Region'],
        var_name='year',
        value_name='male_population'
    )
    male_pop_long['year'] = pd.to_numeric(male_pop_long['year'], errors='coerce')
    return male_pop_long


def _aggregate_population(pop_long):
    """Aggregate male population for age groups 15–49 and 15–59."""
    pop_1559 = pop_long[
        pop_long['age'].isin(AGEGROUPS_15_59) &
        pop_long['year'].isin([2022, 2023])
    ].dropna().groupby(['Region', 'year'], as_index=False)['male_population'].sum()

    pop_1549 = pop_long[
        pop_long['age'].isin(AGEGROUPS_15_49) &
        pop_long['year'].isin([2022, 2023])
    ].dropna().groupby(['Region', 'year'], as_index=False)['male_population'].sum()

    merged = pop_1559.merge(pop_1549, on=['Region', 'year'], how='left')
    merged.columns = ['Region', 'Year', 'male_pop1559', 'male_pop1549']
    return merged


def _normalize_region_names(df, mapping):
    """Replace region names with standardized equivalents."""
    df['Region'] = df['Region'].replace(mapping)
    return df


def compute_normalized_metrics(excess_df, pop_df):
    """Merge mortality and population data, then compute per 100k rates."""
    df_merged = pop_df.merge(excess_df, on=['Region', 'Year'], how='left')
    df_merged = df_merged.dropna()

    df_merged['ex_per100k_1549'] = df_merged['TotalExcess_15_49'] / df_merged['male_pop1549'] * 100_000
    df_merged['ex_per100k_1559'] = df_merged['TotalExcess_15_59'] / df_merged['male_pop1559'] * 100_000

    return df_merged


def main():
    """Run full pipeline to compute and normalize excess male mortality."""
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
        'Чукотский автономный округ': 'Чукотский АО',
    }

    pop_df = _normalize_region_names(pop_df, region_mapping)
    final_df = compute_normalized_metrics(excess_df, pop_df)

    final_df.to_csv("../data/excess_male_mortality_per100k.csv", index=False)


if __name__ == "__main__":
    
    main()
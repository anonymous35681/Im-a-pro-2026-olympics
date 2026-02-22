import pandas as pd

# Set pandas options to display all columns if needed
pd.set_option("display.max_columns", None)


def main():
    file_path = "data/origin_dataset.csv"
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    cols = df.columns

    col_age = cols[5]
    col_edu = cols[6]
    col_income = cols[13]
    col_city_district = cols[1]
    col_city_name = cols[2]
    # col_settlement_type = cols[3] # Unused

    trust_map = {
        "Телевидение": "[Телевидение].1",
        "Интернет-СМИ": "[Интернет-издания].1",
        "Социальные сети": "[Социальные сети].1",
        "Друзья": "[Друзья].1",
        "Газеты": "[Газеты].1",
        "Радио": "[Радио].1",
    }

    col_general_trust_people = cols[14]
    col_trust_surroundings = cols[15]

    col_freq_perception = cols[48]
    col_encounter_freq = cols[43]
    col_where_seen = [cols[44], cols[45], cols[46], cols[47]]
    col_verify = cols[49]
    col_believed = cols[51]
    col_spread = cols[52]

    report_lines = []
    report_lines.append("# Отчет по исследованию датасета")
    report_lines.append("")

    # --- 1. Trust in Sources ---
    report_lines.append("## - Доверие к источникам информации")

    total = len(df)

    media_cols = []
    for label, col_name in trust_map.items():
        if col_name in df.columns:
            trust_count = (
                df[col_name]
                .astype(str)
                .apply(lambda x: 1 if "Доверяю" in x and "Не доверяю" not in x else 0)
                .sum()
            )
            distrust_count = (
                df[col_name]
                .astype(str)
                .apply(lambda x: 1 if "Не доверяю" in x else 0)
                .sum()
            )

            trust_pct = (trust_count / total) * 100
            distrust_pct = (distrust_count / total) * 100
            report_lines.append(
                f"- {label}: Доверяют {trust_pct:.1f}%, Не доверяют {distrust_pct:.1f}%"
            )

            # Collect media columns (excluding "Друзья")
            if label != "Друзья":
                media_cols.append(col_name)
        else:
            report_lines.append(f"- {label}: Нет данных (колонка не найдена)")

    # Calculate overall trust/distrust - % of people who trust/distrust ANY media source
    if media_cols:
        # Person trusts media if they trust at least one source
        any_trust = (
            df[media_cols]
            .astype(str)
            .apply(
                lambda row: any(
                    "Доверяю" in val and "Не доверяю" not in val for val in row
                ),
                axis=1,
            )
            .sum()
        )

        # Person distrusts media if they distrust at least one source
        any_distrust = (
            df[media_cols]
            .astype(str)
            .apply(lambda row: any("Не доверяю" in val for val in row), axis=1)
            .sum()
        )

        overall_trust_pct = (any_trust / total) * 100
        overall_distrust_pct = (any_distrust / total) * 100
        report_lines.append(
            f"- Общее по всем СМИ: Доверяют {overall_trust_pct:.1f}%, Не доверяют {overall_distrust_pct:.1f}%"
        )

    report_lines.append("")

    # --- 2. Frequency of Fakes (Perception) ---
    report_lines.append("## 2) Частота появления фейков (мнение респондентов)")

    if col_freq_perception:
        counts = df[col_freq_perception].value_counts(normalize=True) * 100

        often_sum = 0
        rare_sum = 0
        unsure_sum = 0
        other_sum = 0

        for val, pct in counts.items():
            val_str = str(val).lower()
            match val_str:
                case s if "затрудняюсь" in s:
                    unsure_sum += pct
                case s if "реже" in s:
                    rare_sum += pct
                case s if "чаще" in s:
                    often_sum += pct
                case _:
                    other_sum += pct

        report_lines.append(f"- Считают, что часто (Чаще): {often_sum:.1f}%")
        report_lines.append(f"- Считают, что нечасто (Реже): {rare_sum:.1f}%")
        report_lines.append(f"- Затруднились ответить: {unsure_sum:.1f}%")
        if other_sum > 1:
            report_lines.append(f"- Другое (в т.ч. 'Так же'): {other_sum:.1f}%")

    report_lines.append("")

    # --- 3. Age 45-65+ vs 18-34 Fake Encounters ---
    report_lines.append("## 3) Столкновение с фейками по возрастам")

    df["Age_Clean"] = pd.to_numeric(df[col_age], errors="coerce")

    group_older = df[(df["Age_Clean"] >= 45)]
    group_younger = df[(df["Age_Clean"] >= 18) & (df["Age_Clean"] <= 34)]

    def calc_freq_stats(sub_df, label):
        total_sub = len(sub_df)
        if total_sub == 0:
            return f"{label}: Нет респондентов"

        monthly_plus = (
            sub_df[col_encounter_freq]
            .astype(str)
            .apply(
                lambda x: (
                    1
                    if any(
                        k in x.lower()
                        for k in ["месяц", "неделю", "ежедневно", "каждый день"]
                    )
                    else 0
                )
            )
            .sum()
        )

        daily = (
            sub_df[col_encounter_freq]
            .astype(str)
            .apply(
                lambda x: (
                    1
                    if any(k in x.lower() for k in ["ежедневно", "каждый день"])
                    else 0
                )
            )
            .sum()
        )

        return f"- {label}: Не реже неск. раз в месяц - {(monthly_plus / total_sub) * 100:.1f}%, Почти каждый день - {(daily / total_sub) * 100:.1f}%"

    report_lines.append(calc_freq_stats(group_older, "Люди 45+ лет"))
    report_lines.append(calc_freq_stats(group_younger, "Люди 18-34 лет"))

    report_lines.append("")

    # --- 4. Believed Fakes ---
    report_lines.append("## 4) Вера в фейки")

    counts = df[col_believed].value_counts(normalize=True) * 100

    true_pct = 0
    almost_true_pct = 0

    for val, pct in counts.items():
        val_str = str(val).lower()
        match val_str:
            case s if "безусловно да" in s:
                true_pct += pct
            case s if "скорее да" in s:
                almost_true_pct += pct
            case _:
                pass

    report_lines.append(f"- Приняли за ПРАВДУ (Безусловно да): {true_pct:.1f}%")
    report_lines.append(
        f"- Приняли за ПОЧТИ правду (Скорее да): {almost_true_pct:.1f}%"
    )

    report_lines.append("")

    # --- 5. Demographics of 18-34 ---
    report_lines.append("## 5) Группа риска 18-34")

    group_18_34 = df[(df["Age_Clean"] >= 18) & (df["Age_Clean"] <= 34)]
    total_18_34 = len(group_18_34)

    if total_18_34 > 0:
        believed_counts = (
            group_18_34[col_believed]
            .astype(str)
            .apply(lambda x: 1 if "да" in x.lower() else 0)
            .sum()
        )
        believed_pct = (believed_counts / total_18_34) * 100
        report_lines.append(
            f"- 18-34 лет: Верят фейкам (Безусловно+Скорее да): {believed_pct:.1f}%"
        )

        spread_counts = (
            group_18_34[col_spread]
            .astype(str)
            .apply(lambda x: 1 if "да" in x.lower() else 0)
            .sum()
        )
        spread_pct = (spread_counts / total_18_34) * 100
        report_lines.append(
            f"- Активно распространяют (Приходилось распространять): {spread_pct:.1f}%"
        )

        with_higher = group_18_34[
            group_18_34[col_edu]
            .astype(str)
            .str.contains("высшее", case=False, na=False)
        ]
        without_higher = group_18_34[
            ~group_18_34[col_edu]
            .astype(str)
            .str.contains("высшее", case=False, na=False)
        ]

        def get_belief_rate(sub):
            if len(sub) == 0:
                return 0
            return (
                sub[col_believed]
                .astype(str)
                .apply(lambda x: 1 if "да" in x.lower() else 0)
                .sum()
                / len(sub)
            ) * 100

        report_lines.append(
            f"- Вера в фейки (с высшим): {get_belief_rate(with_higher):.1f}%"
        )
        report_lines.append(
            f"- Вера в фейки (без высшего): {get_belief_rate(without_higher):.1f}%"
        )

        v_counts_all = df[col_verify].value_counts(normalize=True) * 100

        pct_opportunity = 0
        pct_sometimes = 0
        pct_never = 0
        pct_hard = 0
        pct_always = 0

        for val, pct in v_counts_all.items():
            val_str = str(val).lower()
            match val_str:
                case s if "возможност" in s:
                    pct_opportunity += pct
                case s if (
                    "время от времени" in s
                    or "иногда" in s
                    or "зависит от ситуации" in s
                ):
                    pct_sometimes += pct
                case s if "никогда" in s:
                    pct_never += pct
                case s if "затрудняюсь" in s:
                    pct_hard += pct
                case s if "обязательно" in s:
                    pct_always += pct
                case _:
                    pass

        report_lines.append(f"- Жители (Все): Обязательно проверяют: {pct_always:.1f}%")
        report_lines.append(
            f"- Жители (Все): Проверяют по мере возможностей: {pct_opportunity:.1f}%"
        )
        report_lines.append(
            f"- Жители (Все): Проверяют время от времени: {pct_sometimes:.1f}%"
        )
        report_lines.append(
            f"- Жители (Все): Принципиально не проверяют: {pct_never:.1f}%"
        )
        report_lines.append(f"- Жители (Все): Затруднились ответить: {pct_hard:.1f}%")

    report_lines.append("")

    # --- 6. Where Fakes Seen & Risk Groups ---
    report_lines.append("## 6) Где встречали фейки и группы риска")

    seen_cols = col_where_seen

    def check_source_seen(row, keyword):
        for c in seen_cols:
            if pd.notna(row[c]) and keyword.lower() in str(row[c]).lower():
                return True
        return False

    seen_socials = df.apply(
        lambda row: check_source_seen(row, "социальные сети"), axis=1
    )
    seen_internet = df.apply(lambda row: check_source_seen(row, "интернет"), axis=1)
    seen_tv = df.apply(lambda row: check_source_seen(row, "телевидение"), axis=1)

    report_lines.append(
        f"- В социальных сетях: {(seen_socials.sum() / total) * 100:.1f}%"
    )
    report_lines.append(f"- В интернет-СМИ: {(seen_internet.sum() / total) * 100:.1f}%")
    report_lines.append(f"- На телевидении: {(seen_tv.sum() / total) * 100:.1f}%")

    internet_group = df[seen_internet]

    def classify_income(val):
        val_str = str(val).lower()
        match val_str:
            case s if "едва" in s or "питание" in s:
                return "Low"
            case s if "одежду" in s:
                return "Medium"
            case s if "технику" in s or "авто" in s or "ни в чем" in s:
                return "High"
            case _:
                return "Unknown"

    if len(internet_group) > 0:
        internet_capital_pct = (
            internet_group[col_city_district]
            .astype(str)
            .str.contains("Саранск", case=False)
            .sum()
            / len(internet_group)
        ) * 100
        if internet_capital_pct < 1:
            internet_capital_pct = (
                internet_group[col_city_name]
                .astype(str)
                .str.contains("Саранск", case=False)
                .sum()
                / len(internet_group)
            ) * 100

        internet_income_counts = (
            internet_group[col_income]
            .apply(classify_income)
            .value_counts(normalize=True)
            * 100
        )
        report_lines.append(
            f"- Интернет-издания (аудитория): Жители Саранска {internet_capital_pct:.1f}%, Доход средний/высокий {(internet_income_counts.get('Medium', 0) + internet_income_counts.get('High', 0)):.1f}%"
        )

    tv_skeptics = df[
        df["[Телевидение].1"]
        .astype(str)
        .str.contains("Не доверяю", case=False, na=False)
    ]

    if len(tv_skeptics) > 0:
        pct_18_24 = (
            tv_skeptics[
                (tv_skeptics["Age_Clean"] >= 18) & (tv_skeptics["Age_Clean"] <= 24)
            ].shape[0]
            / len(tv_skeptics)
        ) * 100

        pct_saransk = (
            tv_skeptics[col_city_district]
            .astype(str)
            .str.contains("Саранск", case=False)
            .sum()
            / len(tv_skeptics)
        ) * 100
        if pct_saransk < 1:
            pct_saransk = (
                tv_skeptics[col_city_name]
                .astype(str)
                .str.contains("Саранск", case=False)
                .sum()
                / len(tv_skeptics)
            ) * 100

        pct_high_income = (
            tv_skeptics[col_income].apply(classify_income).isin(["High"]).sum()
            / len(tv_skeptics)
        ) * 100

        report_lines.append(
            f"- Скептики ТВ: 18-24 лет {pct_18_24:.1f}%, Саранск {pct_saransk:.1f}%, Высокий доход {pct_high_income:.1f}%"
        )

    # Check for fakes in conversations with friends/relatives
    seen_friends_count = 0
    for col in seen_cols:
        seen_friends_count += (
            df[col]
            .astype(str)
            .str.contains("Друзья, родные, знакомые", case=False, na=False)
            .sum()
        )
    seen_friends_pct = (seen_friends_count / total) * 100
    report_lines.append(
        f"- Сталкивались с фейками в разговорах с друзьями/родственниками: {seen_friends_pct:.1f}%"
    )

    trust_close_circle_col = col_trust_surroundings
    if trust_close_circle_col in df.columns:
        trust_close_circle_pct = (
            df[trust_close_circle_col]
            .astype(str)
            .str.contains("Большинству людей можно доверять", case=False)
            .sum()
            / total
        ) * 100
        report_lines.append(
            f"- Доверяют своим родным и близким (Q15): {trust_close_circle_pct:.1f}%"
        )

    if col_general_trust_people in df.columns:
        general_trust_pct = (
            df[col_general_trust_people]
            .astype(str)
            .str.contains("Большинству людей можно доверять", case=False)
            .sum()
            / total
        ) * 100
        report_lines.append(
            f"- Общий индекс доверия к людям (Q14): {general_trust_pct:.1f}%"
        )

    with open("report.md", "w") as f:
        f.write("\n".join(report_lines))

    print("Analysis complete. Report saved to ./report.md")


if __name__ == "__main__":
    main()

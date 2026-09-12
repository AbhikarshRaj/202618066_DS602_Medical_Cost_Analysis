import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import shapiro, levene, mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.outliers_influence import variance_inflation_factor
# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Analysis",
    layout="wide"
)

# -----------------------------
# Load dataset
# -----------------------------
from pathlib import Path

DATA_PATH = Path(__file__).parent / "Data" / "insurance.csv"

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Title
# -----------------------------
st.title("Medical Insurance Cost Analysis")

st.write(
    "Statistical analysis and prediction of medical insurance charges."
)

# -----------------------------
# Create the required 3 tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "Data Exploration",
    "Hypothesis Testing",
    "Prediction & Diagnostics"
])


# -----------------------------
# TAB 1
# -----------------------------
# -----------------------------
# TAB 1 - DATA EXPLORATION
# -----------------------------
with tab1:

    st.header("Data Exploration")

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.header("Data Filters")

    age_range = st.sidebar.slider(
        "Age Range",
        int(df["age"].min()),
        int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max()))
    )

    bmi_range = st.sidebar.slider(
        "BMI Range",
        float(df["bmi"].min()),
        float(df["bmi"].max()),
        (float(df["bmi"].min()), float(df["bmi"].max()))
    )

    smoker_filter = st.sidebar.selectbox(
        "Smoking Status",
        ["All", "yes", "no"]
    )

    region_filter = st.sidebar.multiselect(
        "Region",
        options=df["region"].unique(),
        default=df["region"].unique()
    )


    # -----------------------------
    # Apply Filters
    # -----------------------------
    filtered_df = df[
        (df["age"] >= age_range[0]) &
        (df["age"] <= age_range[1]) &
        (df["bmi"] >= bmi_range[0]) &
        (df["bmi"] <= bmi_range[1]) &
        (df["region"].isin(region_filter))
    ]

    if smoker_filter != "All":
        filtered_df = filtered_df[
            filtered_df["smoker"] == smoker_filter
        ]


    # -----------------------------
    # Summary Metrics
    # -----------------------------
    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Records",
        len(filtered_df)
    )

    col2.metric(
        "Average Age",
        f"{filtered_df['age'].mean():.1f}"
    )

    col3.metric(
        "Average BMI",
        f"{filtered_df['bmi'].mean():.2f}"
    )

    col4.metric(
        "Average Charges",
        f"${filtered_df['charges'].mean():,.2f}"
    )


    # -----------------------------
    # Dataset Preview
    # -----------------------------
    st.subheader("Filtered Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


    # -----------------------------
    # Descriptive Statistics
    # -----------------------------
    st.subheader("Descriptive Statistics")

    st.dataframe(
        filtered_df.describe(),
        use_container_width=True
    )


    # -----------------------------
    # Charges Histogram
    # -----------------------------
    st.subheader("Distribution of Medical Charges")

    fig1 = px.histogram(
        filtered_df,
        x="charges",
        nbins=30,
        title="Distribution of Medical Insurance Charges"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


    # -----------------------------
    # Age vs Charges
    # -----------------------------
    st.subheader("Age vs Medical Charges")

    fig2 = px.scatter(
        filtered_df,
        x="age",
        y="charges",
        color="smoker",
        hover_data=["sex", "bmi", "children", "region"],
        title="Age vs Medical Charges by Smoking Status"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )
    # -----------------------------
    # BMI Distribution
    # -----------------------------
    st.subheader("Distribution of BMI")

    fig3 = px.histogram(
        filtered_df,
        x="bmi",
        nbins=30,
        title="Distribution of BMI"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


    # -----------------------------
    # BMI vs Charges
    # -----------------------------
    st.subheader("BMI vs Medical Charges")

    fig4 = px.scatter(
        filtered_df,
        x="bmi",
        y="charges",
        color="smoker",
        hover_data=["age", "sex", "children", "region"],
        title="BMI vs Medical Charges by Smoking Status"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


    # -----------------------------
    # Smokers vs Non-Smokers
    # -----------------------------
    st.subheader("Medical Charges by Smoking Status")

    fig5 = px.box(
        filtered_df,
        x="smoker",
        y="charges",
        title="Smokers vs Non-Smokers Medical Charges"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )


    # -----------------------------
    # Charges by Region
    # -----------------------------
    st.subheader("Medical Charges by Region")

    fig6 = px.box(
        filtered_df,
        x="region",
        y="charges",
        title="Medical Charges Across Regions"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )


    # -----------------------------
    # Correlation Matrix
    # -----------------------------
    st.subheader("Correlation Matrix")

    numeric_df = filtered_df[
        ["age", "bmi", "children", "charges"]
    ]

    correlation = numeric_df.corr()

    fig7 = px.imshow(
        correlation,
        text_auto=".2f",
        title="Correlation Matrix of Numerical Variables"
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )


# -----------------------------
# TAB 2 - HYPOTHESIS TESTING
# -----------------------------
with tab2:

    st.header("Hypothesis Testing Lab")

    test_choice = st.selectbox(
        "Choose a Hypothesis Test",
        [
            "Smokers vs Non-Smokers",
            "Medical Charges Across Regions"
        ]
    )

    alpha = 0.05

    # =========================================================
    # TEST 1: SMOKERS VS NON-SMOKERS
    # =========================================================
    if test_choice == "Smokers vs Non-Smokers":

        st.subheader("Smokers vs Non-Smokers Medical Charges")

        st.markdown("""
        **H0:** There is no significant difference in medical charges between smokers and non-smokers.  
        **H1:** There is a significant difference in medical charges between smokers and non-smokers.
        """)

        smoker_charges = df[df["smoker"] == "yes"]["charges"]
        nonsmoker_charges = df[df["smoker"] == "no"]["charges"]

        # -----------------------------
        # Shapiro-Wilk Normality Test
        # -----------------------------
        smoker_shapiro = shapiro(smoker_charges)
        nonsmoker_shapiro = shapiro(nonsmoker_charges)

        st.subheader("Normality Check (Shapiro-Wilk)")

        col1, col2 = st.columns(2)

        col1.metric(
            "Smoker p-value",
            f"{smoker_shapiro.pvalue:.4e}"
        )

        col2.metric(
            "Non-Smoker p-value",
            f"{nonsmoker_shapiro.pvalue:.4e}"
        )

        if smoker_shapiro.pvalue < alpha or nonsmoker_shapiro.pvalue < alpha:
            st.warning("Normality assumption is not satisfied.")
        else:
            st.success("Normality assumption is satisfied.")

        # -----------------------------
        # Levene's Test
        # -----------------------------
        levene_result = levene(smoker_charges, nonsmoker_charges)

        st.subheader("Equal Variance Check (Levene's Test)")

        st.metric(
            "Levene p-value",
            f"{levene_result.pvalue:.4e}"
        )

        if levene_result.pvalue < alpha:
            st.warning("Equal variance assumption is not satisfied.")
        else:
            st.success("Equal variance assumption is satisfied.")

        # -----------------------------
        # Mann-Whitney U Test
        # -----------------------------
        result = mannwhitneyu(
            smoker_charges,
            nonsmoker_charges,
            alternative="two-sided"
        )

        st.subheader("Final Test Used: Mann-Whitney U Test")

        col1, col2 = st.columns(2)

        col1.metric(
            "U Statistic",
            f"{result.statistic:.2f}"
        )

        col2.metric(
            "p-value",
            f"{result.pvalue:.4e}"
        )

        # -----------------------------
        # Visual Comparison
        # -----------------------------
        st.subheader("Visual Comparison")

        fig_test1 = px.box(
            df,
            x="smoker",
            y="charges",
            points="outliers",
            title="Medical Charges: Smokers vs Non-Smokers"
        )

        st.plotly_chart(
            fig_test1,
            use_container_width=True
        )

        # -----------------------------
        # Final Decision
        # -----------------------------
        st.subheader("Final Decision")

        if result.pvalue < alpha:
            st.success(
                "Reject H0: Medical charges differ significantly between smokers and non-smokers."
            )
        else:
            st.warning(
                "Fail to Reject H0: There is not enough evidence of a significant difference."
            )

    # =========================================================
    # TEST 2: MEDICAL CHARGES ACROSS REGIONS
    # =========================================================
    elif test_choice == "Medical Charges Across Regions":

        st.subheader("Medical Charges Across Regions")

        st.markdown("""
        **H0:** Medical charge distributions do not significantly differ across regions.  
        **H1:** At least one region differs significantly.
        """)

        northeast = df[df["region"] == "northeast"]["charges"]
        northwest = df[df["region"] == "northwest"]["charges"]
        southeast = df[df["region"] == "southeast"]["charges"]
        southwest = df[df["region"] == "southwest"]["charges"]

        # -----------------------------
        # Kruskal-Wallis Test
        # -----------------------------
        result = kruskal(
            northeast,
            northwest,
            southeast,
            southwest
        )

        st.subheader("Final Test Used: Kruskal-Wallis Test")

        col1, col2 = st.columns(2)

        col1.metric(
            "H Statistic",
            f"{result.statistic:.3f}"
        )

        col2.metric(
            "p-value",
            f"{result.pvalue:.4f}"
        )

        # -----------------------------
        # Visual Comparison
        # -----------------------------
        st.subheader("Visual Comparison")

        fig_test2 = px.box(
            df,
            x="region",
            y="charges",
            points="outliers",
            title="Medical Charges Across Regions"
        )

        st.plotly_chart(
            fig_test2,
            use_container_width=True
        )

        # -----------------------------
        # Final Decision
        # -----------------------------
        st.subheader("Final Decision")

        if result.pvalue < alpha:
            st.success(
                "Reject H0: Medical charges differ significantly across regions."
            )
        else:
            st.warning(
                "Fail to Reject H0: There is not enough evidence of a significant regional difference."
            )
# -----------------------------
# TAB 3
# -----------------------------
# -----------------------------
# TAB 3 - PREDICTION & DIAGNOSTICS
# -----------------------------
with tab3:

    st.header("Live Prediction & Regression Diagnostics")

    # =========================================================
    # FIT THE MAIN OLS MODEL
    # =========================================================

    model = smf.ols(
        formula="charges ~ age + sex + bmi + children + smoker + region",
        data=df
    ).fit()


    # =========================================================
    # MODEL PERFORMANCE
    # =========================================================

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "R-squared",
        f"{model.rsquared:.3f}"
    )

    col2.metric(
        "Adjusted R-squared",
        f"{model.rsquared_adj:.3f}"
    )

    col3.metric(
        "F-statistic",
        f"{model.fvalue:.2f}"
    )

    st.write(
        "The model explains approximately "
        f"{model.rsquared * 100:.1f}% of the variation in medical charges."
    )


    # =========================================================
    # LIVE USER INPUT
    # =========================================================

    st.subheader("Predict Medical Insurance Charges")

    st.write(
        "Enter the patient characteristics below to generate a prediction."
    )

    col1, col2 = st.columns(2)


    # -----------------------------
    # LEFT COLUMN
    # -----------------------------
    with col1:

        age_input = st.number_input(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=30,
            step=1
        )

        bmi_input = st.number_input(
            "BMI",
            min_value=float(df["bmi"].min()),
            max_value=float(df["bmi"].max()),
            value=25.0,
            step=0.1
        )

        children_input = st.number_input(
            "Number of Children",
            min_value=int(df["children"].min()),
            max_value=int(df["children"].max()),
            value=0,
            step=1
        )


    # -----------------------------
    # RIGHT COLUMN
    # -----------------------------
    with col2:

        sex_input = st.selectbox(
            "Sex",
            ["female", "male"]
        )

        smoker_input = st.selectbox(
            "Smoking Status",
            ["no", "yes"]
        )

        region_input = st.selectbox(
            "Region",
            [
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ]
        )


    # =========================================================
    # CREATE DATA FOR PREDICTION
    # =========================================================

    new_person = pd.DataFrame({
        "age": [age_input],
        "sex": [sex_input],
        "bmi": [bmi_input],
        "children": [children_input],
        "smoker": [smoker_input],
        "region": [region_input]
    })


    # =========================================================
    # PREDICTION BUTTON
    # =========================================================

    if st.button("Predict Medical Charges"):

        prediction = model.get_prediction(new_person)

        prediction_summary = prediction.summary_frame(alpha=0.05)

        predicted_charge = prediction_summary["mean"].iloc[0]

        mean_lower = prediction_summary["mean_ci_lower"].iloc[0]
        mean_upper = prediction_summary["mean_ci_upper"].iloc[0]

        prediction_lower = prediction_summary["obs_ci_lower"].iloc[0]
        prediction_upper = prediction_summary["obs_ci_upper"].iloc[0]


        # -----------------------------
        # Main Prediction
        # -----------------------------
        st.subheader("Prediction Result")

        st.metric(
            "Predicted Medical Charges",
            f"${predicted_charge:,.2f}"
        )


        # -----------------------------
        # Confidence Interval
        # -----------------------------
        st.write("### 95% Confidence Interval")

        st.info(
            f"${mean_lower:,.2f}  to  ${mean_upper:,.2f}"
        )

        st.caption(
            "This interval estimates the average medical charge "
            "for people with similar characteristics."
        )


        # -----------------------------
        # Prediction Interval
        # -----------------------------
        st.write("### 95% Prediction Interval")

        st.info(
            f"${prediction_lower:,.2f}  to  ${prediction_upper:,.2f}"
        )

        st.caption(
            "This wider interval represents the likely range for "
            "an individual person's medical charge."
        )


    # =========================================================
    # MODEL COEFFICIENTS
    # =========================================================

    st.divider()

    st.subheader("Regression Coefficients")

    coefficients = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "p-value": model.pvalues.values
    })

    coefficients["Coefficient"] = coefficients["Coefficient"].round(3)
    coefficients["p-value"] = coefficients["p-value"].round(4)

    st.dataframe(
        coefficients,
        use_container_width=True
    )


    # =========================================================
    # RESIDUAL DIAGNOSTICS
    # =========================================================

    st.divider()

    st.header("Regression Diagnostic Checks")

    fitted_values = model.fittedvalues
    residuals = model.resid


    # =========================================================
    # 1. RESIDUALS VS FITTED
    # =========================================================

    st.subheader("1. Linearity & Homoscedasticity")

    fig_residual, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        fitted_values,
        residuals,
        alpha=0.5
    )

    ax.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    ax.set_title("Residuals vs Fitted Values")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")

    st.pyplot(fig_residual)

    plt.close(fig_residual)

    st.write(
        """
        Ideally, residuals should be randomly scattered around zero.
        Visible patterns suggest possible non-linearity, while changes
        in residual spread suggest heteroscedasticity.
        """
    )


    # =========================================================
    # 2. Q-Q PLOT
    # =========================================================

    st.subheader("2. Normality of Residuals")

    fig_qq = sm.qqplot(
        residuals,
        line="45",
        fit=True
    )

    plt.title("Q-Q Plot of Regression Residuals")

    st.pyplot(fig_qq)

    plt.close(fig_qq)


    # =========================================================
    # JARQUE-BERA / OMNIBUS
    # =========================================================

    jb_stat, jb_pvalue, skew, kurtosis = sm.stats.jarque_bera(
        residuals
    )

    omnibus_stat, omnibus_pvalue = sm.stats.omni_normtest(
        residuals
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Jarque-Bera p-value",
            f"{jb_pvalue:.4e}"
        )

    with col2:

        st.metric(
            "Omnibus p-value",
            f"{omnibus_pvalue:.4e}"
        )


    if jb_pvalue < 0.05:

        st.warning(
            "Residual normality assumption is not satisfied."
        )

    else:

        st.success(
            "There is no strong evidence against residual normality."
        )


    # =========================================================
    # 3. MULTICOLLINEARITY - VIF
    # =========================================================

    st.subheader("3. Multicollinearity - VIF")

    X_vif = df[
        ["age", "bmi", "children"]
    ].copy()

    vif_table = pd.DataFrame()

    vif_table["Variable"] = X_vif.columns

    vif_table["VIF"] = [
        variance_inflation_factor(
            X_vif.values,
            i
        )
        for i in range(X_vif.shape[1])
    ]

    vif_table["VIF"] = vif_table["VIF"].round(3)

    st.dataframe(
        vif_table,
        use_container_width=True
    )


    if (vif_table["VIF"] < 5).all():

        st.success(
            "No serious multicollinearity detected. "
            "All continuous predictors have VIF below 5."
        )

    else:

        st.warning(
            "Some predictors have high VIF values and may "
            "have multicollinearity problems."
        )


    # =========================================================
    # FINAL DIAGNOSTIC SUMMARY
    # =========================================================

    st.subheader("Diagnostic Summary")

    st.write(
        """
        - **Linearity:** Residual patterns should be inspected visually.
        - **Homoscedasticity:** The spread of residuals should remain approximately constant.
        - **Normality:** The Q-Q plot and Jarque-Bera/Omnibus tests indicate whether residuals are normally distributed.
        - **Multicollinearity:** VIF values below 5 indicate no serious multicollinearity problem.
        """
    )
"""
transform_functions.py
=======================
Beginner-annotated schema-alignment functions for the Employee Attrition project.

WHAT THIS FILE DOES
--------------------
It defines two functions:
    transform_ibm(df_ibm)  -> takes the raw IBM HR CSV loaded into a pandas
                               DataFrame, and returns a NEW DataFrame whose
                               columns match our unified schema.
    transform_hrd(df_hrd)  -> does the same thing, but for HRDataset_v14.

Then at the bottom, combine_datasets() stacks both outputs into one table.

WHY TWO SEPARATE FUNCTIONS INSTEAD OF ONE BIG SCRIPT?
------------------------------------------------------
This is a software-engineering habit worth learning early: each function has
ONE job (translate one dataset), which makes it independently testable.
You can call transform_ibm() on a tiny fake DataFrame and check the output
without needing HRDataset_v14 to even exist. That's the core idea behind
"unit testing" - you'll want this later for your CI/CD pipeline (Experiment 7).
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# SHARED HELPER 1: department bucketing
# ---------------------------------------------------------------------------
# WHY A HELPER FUNCTION AT ALL?
# Both transform_ibm() and transform_hrd() need to bucket departments into
# the same 3 categories, but they start from DIFFERENT raw department names.
# Rather than copy-pasting bucketing logic into both functions (which risks
# them drifting out of sync if you edit one and forget the other), we write
# the logic ONCE here and call it from both places. This is the "Don't
# Repeat Yourself" (DRY) principle - one of the most important habits in
# real codebases.

def bucket_department(raw_department: str) -> str:
    """
    Takes a messy, source-specific department string and returns one of:
    'Technical', 'Sales_Admin', or 'Support'.

    HOW THIS FUNCTION WORKS:
    `raw_department` comes in as a single string, e.g. "Research & Development"
    or "IT/IS". We lowercase it so our comparisons aren't broken by
    capitalization differences (e.g. "IT/IS" vs "it/is"), then check which
    keywords it contains using the `in` operator, which for strings checks
    "is this substring present anywhere inside this string?"
    """
    if pd.isna(raw_department):
        # pd.isna() checks for NaN/None/missing values. We handle this FIRST
        # because calling .lower() on a missing value (NaN, a float) would
        # crash the program with an AttributeError. Always guard against
        # missing data before doing string operations.
        return np.nan

    text = raw_department.lower()  # .lower() converts "IT/IS" -> "it/is"

    # `any(...)` returns True if AT LEAST ONE item in the list/generator is True.
    # Here we're checking: does `text` contain ANY of these keywords?
    if any(keyword in text for keyword in ["research", "engineering", "it/is", "software", "production"]):
        return "Technical"
    elif any(keyword in text for keyword in ["sales", "admin", "hr", "human resources"]):
        return "Sales_Admin"
    else:
        return "Support"


# ---------------------------------------------------------------------------
# SHARED HELPER 2: performance rating -> ordinal scale
# ---------------------------------------------------------------------------
# WHY A DICTIONARY LOOKUP INSTEAD OF IF/ELIF?
# When you have a fixed, known set of input values mapping to a fixed set of
# output values (not a "does it contain a keyword" situation), a dictionary
# is the cleaner, faster, more readable tool. This is what .map() below uses.

PERFORMANCE_MAP_HRD = {
    "PIP": "Low",                    # Performance Improvement Plan = lowest
    "Needs Improvement": "Low",
    "Fully Meets": "Good",
    "Exceeds": "Excellent",
}
# IBM's PerformanceRating is numeric (3 or 4). We map those numbers to the
# SAME label set so both sources end up speaking the same ordinal language.
PERFORMANCE_MAP_IBM = {
    1: "Low",
    2: "Good",
    3: "Excellent",
    4: "Outstanding",
}


# ---------------------------------------------------------------------------
# TRANSFORM FUNCTION 1: IBM HR Analytics
# ---------------------------------------------------------------------------

def transform_ibm(df_ibm: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the raw IBM HR Analytics DataFrame into our unified schema.

    PARAMETER:
        df_ibm : the DataFrame you get from pd.read_csv("ibm_hr.csv")

    RETURNS:
        a NEW DataFrame with unified column names, ready to be stacked
        with the output of transform_hrd().
    """

    # We build the new dataset as an empty dictionary first, then convert it
    # to a DataFrame at the end. WHY? Building a dict of {column_name: Series}
    # and calling pd.DataFrame(dict) ONCE is more efficient than repeatedly
    # writing out['col'] = ... on a growing DataFrame (which pandas has to
    # reallocate memory for on every single assignment). For a beginner,
    # the main thing to know: this is just a cleaner, faster pattern.
    out = {}

    # --- Bucket 1: simple copy/rename ---
    # `df_ibm['Age']` pulls out one column as a pandas Series (think of it as
    # a single labeled column of data, like one column in Excel).
    out["age"] = df_ibm["Age"]
    out["monthly_income"] = df_ibm["MonthlyIncome"]
    out["tenure_years"] = df_ibm["YearsAtCompany"]
    out["num_companies_worked"] = df_ibm["NumCompaniesWorked"]
    out["training_times_last_year"] = df_ibm["TrainingTimesLastYear"]
    out["distance_from_home"] = df_ibm["DistanceFromHome"]

    # --- employee_id and source_dataset: identity + provenance tracking ---
    # `.astype(str)` converts the EmployeeNumber column (which is numeric)
    # into text, because we're about to glue a text prefix onto it, and you
    # can't add a string to a number in Python.
    # The `"ibm_" + ...` part uses the `+` operator, which for pandas string
    # Series means "concatenate this text onto the front of every value" -
    # it runs this operation on the WHOLE column at once (this is called
    # "vectorization" - much faster than writing a manual loop).
    out["employee_id"] = "ibm_" + df_ibm["EmployeeNumber"].astype(str)

    # A single fixed value assigned to a dict key. When we build the
    # DataFrame at the end, pandas automatically repeats this value for
    # every row. This is how we "stamp" every IBM row with its origin.
    out["source_dataset"] = "ibm"

    # --- Bucket 2: recode with .map() ---
    # `.map({...})` looks up EVERY value in the column against the dictionary
    # and replaces it with the matching value. Think of it like Excel's
    # VLOOKUP, but applied to an entire column in one line.
    out["gender"] = df_ibm["Gender"]  # already 'Male'/'Female' - no change needed
    out["marital_status"] = df_ibm["MaritalStatus"]  # already clean text
    out["attrition_label"] = df_ibm["Attrition"].map({"Yes": 1, "No": 0})
    out["overtime_flag"] = df_ibm["OverTime"].map({"Yes": 1, "No": 0})

    # --- Bucket 3: compute new values ---
    # `.apply(function)` runs a Python function on EVERY value in the column,
    # one at a time, and collects the results back into a new column.
    # It's slower than a vectorized operation like .map(), but it's the
    # right tool when your logic is too complex for a simple lookup table
    # (like our keyword-based bucket_department() function).
    out["department_category"] = df_ibm["Department"].apply(bucket_department)
    out["performance_rating"] = df_ibm["PerformanceRating"].map(PERFORMANCE_MAP_IBM)

    # Satisfaction score: IBM stores 4 separate 1-4 scales. We average them,
    # then rescale that average from a 1-4 range down to a clean 0-1 range.
    # `df[[...]]` with a LIST of column names selects multiple columns at
    # once (a mini-DataFrame), and `.mean(axis=1)` takes the average ACROSS
    # those columns for each row (axis=1 means "average sideways, across
    # columns", as opposed to axis=0 which would average down a column).
    satisfaction_cols = ["EnvironmentSatisfaction", "JobSatisfaction",
                          "RelationshipSatisfaction", "WorkLifeBalance"]
    raw_avg = df_ibm[satisfaction_cols].mean(axis=1)   # produces values 1.0 - 4.0
    out["satisfaction_score"] = (raw_avg - 1) / (4 - 1)  # rescales 1-4 range to 0-1

    # --- Bucket 4: missing by design (HRDataset-only concepts) ---
    # We explicitly create these columns and fill with pd.NA (pandas' modern
    # "missing value" marker) rather than just leaving them out. WHY?
    # If we skip them entirely, when we later concatenate with transform_hrd()'s
    # output, pandas will auto-create them as NaN anyway - but doing it
    # explicitly here makes the missingness a DELIBERATE, DOCUMENTED decision
    # instead of an accident, and it means both functions produce DataFrames
    # with the exact same set of columns (good practice - always check
    # `set(df1.columns) == set(df2.columns)` before concatenating).
    # NOTE ON DTYPE: we build a full-length Series of pd.NA with an EXPLICIT
    # dtype ("string" for text fields, "Int64" for numeric ones) rather than
    # assigning a single bare `pd.NA`. WHY THIS MATTERS: when we later
    # pd.concat() this against transform_hrd()'s output (where these same
    # columns hold REAL values), pandas needs to know what type an all-missing
    # column "would have been" to combine it correctly with the real one.
    # Leaving it as a bare pd.NA makes pandas guess - and that guess is
    # exactly what changes between pandas versions (the FutureWarning you saw).
    # Declaring the dtype ourselves removes the guesswork entirely.
    n_rows = len(df_ibm)
    out["race_ethnicity"] = pd.Series([pd.NA] * n_rows, dtype="string")
    out["manager_id"] = pd.Series([pd.NA] * n_rows, dtype="string")
    out["recruitment_source"] = pd.Series([pd.NA] * n_rows, dtype="string")
    out["special_projects_count"] = pd.Series([pd.NA] * n_rows, dtype="Int64")

    # Finally: convert our dictionary of columns into an actual DataFrame.
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# TRANSFORM FUNCTION 2: HRDataset_v14
# ---------------------------------------------------------------------------

def transform_hrd(df_hrd: pd.DataFrame, reference_date: str = "2019-01-01") -> pd.DataFrame:
    """
    Converts the raw HRDataset_v14 DataFrame into our unified schema.

    `reference_date` is the "as-of" date we use to compute ages and tenure.
    We make it a PARAMETER (not hardcoded inside the function) so that if you
    later decide to use a different snapshot date, you change one line at
    the call site instead of hunting through the function body. This is a
    general good habit: don't bury important assumptions deep in your code.
    """

    out = {}

    out["employee_id"] = "hrd_" + df_hrd["EmpID"].astype(str)
    out["source_dataset"] = "hrd"

    # --- Cleaning BEFORE recoding: the .str.strip() catch ---
    # HRDataset_v14's `Sex` column has trailing whitespace, e.g. "M " instead
    # of "M". If we tried `.map({"M": "Male", ...})` directly, "M " would NOT
    # match "M" in our dictionary (whitespace matters to Python!) and every
    # row would silently become NaN. This is exactly the kind of bug that
    # doesn't crash your program - it just quietly corrupts your data - which
    # is why "always inspect a few unique values before mapping" is a habit
    # worth building now.
    # `.str.strip()` removes leading/trailing whitespace from every value in
    # the column (the `.str` prefix tells pandas "treat this column as text
    # and apply this text operation to every entry").
    cleaned_sex = df_hrd["Sex"].str.strip()
    out["gender"] = cleaned_sex.map({"M": "Male", "F": "Female"})

    out["marital_status"] = df_hrd["MaritalDesc"]
    out["monthly_income"] = df_hrd["Salary"]

    # --- Bucket 3 continued: date arithmetic ---
    # pd.to_datetime() converts a column of text dates (like "07/10/83")
    # into an actual datetime type that Python can do MATH on. You cannot
    # subtract two text strings to get "how many days apart are these", but
    # you CAN subtract two datetime objects.
    #
    # WHY WE PASS format= EXPLICITLY:
    # Without a format, pandas has to GUESS how to read each date string,
    # and with ambiguous 2-digit years (e.g. "7/10/83" - is that day=7 or
    # month=7? year 1983 or 2083?) a wrong guess fails silently - you don't
    # get an error, you just get a wrong age for that employee. HRDataset_v14
    # uses M/D/YY format, so we tell pandas that explicitly instead of
    # letting it guess row-by-row.
    dob = pd.to_datetime(df_hrd["DOB"], format="%m/%d/%y", errors="coerce")
    # `errors="coerce"` means: if a date fails to parse (e.g. it's garbled
    # or empty), don't crash the whole program - just put NaT (pandas'
    # "Not a Time" missing-value marker) there instead, and keep going.
    #
    # THE 2-DIGIT-YEAR TRAP: "%y" (lowercase) assumes years 00-68 mean
    # 2000-2068 and years 69-99 mean 1969-1999 (a fixed pandas/Python
    # convention called the "pivot year"). For a workforce dataset this is
    # usually fine (nobody's born in 2071), but ALWAYS verify - see the
    # sanity check below.
    ref = pd.to_datetime(reference_date)

    # Subtracting two datetime Series gives you a "timedelta" (a duration).
    # `.dt.days` pulls out just the number of days from that duration.
    # We divide by 365.25 (accounting for leap years) to convert days to
    # approximate years, then round to the nearest whole number.
    out["age"] = ((ref - dob).dt.days / 365.25).round().astype("Int64")
    # `.astype("Int64")` (capital I) is pandas' "nullable integer" type -
    # unlike Python's normal int, it can hold NaN values for rows where the
    # date failed to parse, instead of crashing.

    # SANITY CHECK - always run this after date parsing, don't just trust it.
    # A working-age person should be roughly 18-75. If pandas mis-parsed the
    # century (the 2-digit-year trap mentioned above), you'd see ages that
    # are negative or absurdly large here, and you'd rather catch that now
    # than three notebooks later when a model quietly trains on garbage ages.
    bad_ages = out["age"][(out["age"] < 16) | (out["age"] > 80)]
    if len(bad_ages) > 0:
        print(f"WARNING: {len(bad_ages)} HRDataset rows have implausible ages "
              f"after DOB parsing - inspect these before trusting the output.")

    date_of_hire = pd.to_datetime(df_hrd["DateofHire"], format="%m/%d/%y", errors="coerce")
    out["tenure_years"] = ((ref - date_of_hire).dt.days / 365.25).round().astype("Int64")

    # --- attrition_label with a business-logic exclusion ---
    # `Termd` is already 0/1, so on the surface this looks like a plain copy.
    # But per our schema-mapping decision, we only want VOLUNTARY attrition
    # to count as a positive "at risk of leaving" label. `TermReason`
    # contains free text like "career change" or "layoff".
    # `.str.contains("layoff|downsizing", case=False, na=False)` checks each
    # row's TermReason for either word (the `|` means "OR" in this pattern-
    # matching syntax, called a "regular expression"). `case=False` makes it
    # ignore uppercase/lowercase differences. `na=False` means "if TermReason
    # is missing, treat that as NOT matching" (rather than crashing/erroring).
    is_involuntary = df_hrd["TermReason"].str.contains(
        "layoff|downsizing", case=False, na=False
    )
    attrition_raw = df_hrd["Termd"]
    # `.where(condition, other)` keeps the original value where `condition`
    # is True, and replaces it with `other` where `condition` is False.
    # Here: keep attrition_raw as-is UNLESS it was an involuntary exit, in
    # which case we blank it out to NaN (so it's excluded from training,
    # rather than wrongly counted as a "voluntary risk" case).
    out["attrition_label"] = attrition_raw.where(~is_involuntary, np.nan)
    # The `~` symbol means "NOT" - it flips True to False and vice versa.

    out["department_category"] = df_hrd["Department"].apply(bucket_department)
    out["performance_rating"] = df_hrd["PerformanceScore"].map(PERFORMANCE_MAP_HRD)

    # Two satisfaction-ish fields on different scales (1-5), averaged then
    # rescaled to 0-1, same logic as the IBM function above.
    raw_avg = df_hrd[["EngagementSurvey", "EmpSatisfaction"]].mean(axis=1)
    out["satisfaction_score"] = (raw_avg - 1) / (5 - 1)

    out["race_ethnicity"] = df_hrd["RaceDesc"]
    out["manager_id"] = df_hrd["ManagerID"]
    out["recruitment_source"] = df_hrd["RecruitmentSource"]
    out["special_projects_count"] = df_hrd["SpecialProjectsCount"]

    # IBM-only concepts: explicitly missing here, for the same reason as above -
    # explicit dtype instead of a bare pd.NA (see the note in transform_ibm()).
    n_rows = len(df_hrd)
    out["num_companies_worked"] = pd.Series([pd.NA] * n_rows, dtype="Int64")
    out["training_times_last_year"] = pd.Series([pd.NA] * n_rows, dtype="Int64")
    out["overtime_flag"] = pd.Series([pd.NA] * n_rows, dtype="Int64")
    out["distance_from_home"] = pd.Series([pd.NA] * n_rows, dtype="Int64")

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# COMBINING STEP
# ---------------------------------------------------------------------------

def combine_datasets(df_ibm_raw: pd.DataFrame, df_hrd_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Runs both translators and stacks their output into one unified table.
    """
    ibm_clean = transform_ibm(df_ibm_raw)
    hrd_clean = transform_hrd(df_hrd_raw)

    # A SAFETY CHECK worth learning as a habit: before stacking two
    # DataFrames, confirm they actually have the same columns. `set(...)`
    # converts the column list into a set (an unordered collection with no
    # duplicates), and comparing two sets with `==` tells you if they contain
    # exactly the same items, regardless of order.
    assert set(ibm_clean.columns) == set(hrd_clean.columns), \
        "Schema mismatch! transform_ibm() and transform_hrd() must output identical columns."

    # pd.concat([...]) stacks DataFrames on top of each other (like stacking
    # two spreadsheets that have the same column headers). `ignore_index=True`
    # tells pandas to renumber the rows 0, 1, 2, ... instead of keeping the
    # original row numbers from each source (which would otherwise create
    # duplicate index labels like two different rows both being "row 0").
    combined = pd.concat([ibm_clean, hrd_clean], ignore_index=True)
    return combined


if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # (`python transform_functions.py`), NOT when another script imports
    # these functions with `from transform_functions import transform_ibm`.
    # It's a common Python pattern for "quick manual test when I run this
    # file on its own."
    df_ibm_raw = pd.read_csv(r"D:\Shin\Programming\ADS_project\data\raw\IBM_HR_Attrition.csv")
    df_hrd_raw = pd.read_csv(r"D:\Shin\Programming\ADS_project\data\raw\HRDataset_v14.csv")

    combined = combine_datasets(df_ibm_raw, df_hrd_raw)
    print(combined.shape)
    print(combined["source_dataset"].value_counts())
    combined.to_csv(r"D:\Shin\Programming\ADS_project\data\processed\combined_unified.csv", index=False)

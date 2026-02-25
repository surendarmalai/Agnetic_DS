import os
import re
from langchain_groq import ChatGroq
from source_code.state import AgentState

# ─────────────────────────────────────────────
#  TELECOM DOMAIN KNOWLEDGE BASES
# ─────────────────────────────────────────────

TELECOM_ABBREVIATION_MAP = """
Known Telecom Abbreviations & Their Meanings:
- aon         → AgeOnNetwork
- mou         → MinutesOfUse
- arpu        → AvgRevenuePerUser
- rev         → Revenue
- cnt         → Count
- amt         → Amount
- vol         → Volume
- avg         → Avg
- msisdn      → Msisdn (treat as ID)
- imsi        → Imsi (treat as ID)
- cust        → Customer
- acct        → Account
- seg         → Segment
- flg / flag  → Flag
- ind         → Flag (binary indicator)
- dt          → Date
- mnth / mth  → Month
- wk          → Week
- l30d        → Last30d
- l60d        → Last60d
- l90d        → Last90d
- p30d        → Prev30d
- rchg        → Recharge
- tot / ttl   → Total
- outg        → Outgoing
- inc / incmg → Incoming
- intl        → Intl
- roam        → Roaming
- vas         → Vas
- gprs        → Data
- offnet      → Offnet
- onnet       → Onnet
- clv / ltv   → Ltv
- comp        → Complaint
- cc          → CustCare
- gndr        → Gender
- rgn         → Region
- actv        → Activation
- deactv      → Deactivation
- dob         → BirthDate
- hva         → HighValue
- mva         → MidValue
- lva         → LowValue
"""

TELECOM_DATA_UNIT_RULES = """
Data Unit Detection Rules:

NEVER append units to these — the unit is implied by the metric name:
- Any Mou column → unit (minutes) is implicit, no suffix
- AgeOnNetwork → days is the telecom default, no suffix
- Arpu, Revenue, Recharge amounts → currency is implied, no suffix

ALWAYS append units to these — genuinely ambiguous without it:
- Data/Gprs/Vol columns: infer from max value in df.describe
    max > 1,000,000  → Bytes
    max 10k–1,000,000 → Kb
    max 1,000–10,000  → Mb
    max < 1,000       → Gb
- Non-standard duration columns (not Mou): append Mins or Secs based on magnitude
"""

TELECOM_DATE_DETECTION_RULES = """
Date Field Detection Rules:
- Integer columns whose name contains: date, dt, dob, join, activ, deactiv, birth
  are almost always integers in YYYYMMDD format (e.g., 20250516)
- 8-digit integers between 19000101 and 20991231 → YYYYMMDD date
- 6-digit integers → YYYYMM format
- Rename to clearly indicate date nature e.g. ActivationDate, BirthDate
- Do NOT convert — renaming only.
"""

STANDARD_CHURN_SCHEMA = """
Standard Churn Model Target Schema — use these names where a match exists:
IDENTIFIERS:  CustomerId, Msisdn, AccountId
DEMOGRAPHICS: AgeYears, Gender, Region, City, Nationality, MaritalStatus, Segment
TENURE:       TenureMonths, TenureDays, ActivationDate, DeactivationDate, ContractType
REVENUE:      ArpuMonthly, RevenueLast30d, RevenueLast60d, RevenueLast90d, TotalRevenue
VOICE:        MouOutgoingLast30dMins, MouIncomingLast30dMins, MouOffnetLast30dMins,
              MouOnnetLast30dMins, MouIntlLast30dMins
DATA:         DataUsageLast30dKb, DataUsageLast60dKb, DataRechargeCountLast30d
SMS:          SmsOutgoingLast30d, SmsIncomingLast30d
RECHARGES:    RechargeAmountLast30d, RechargeCountLast30d, DaysSinceLastRecharge
VAS:          VasSubCount, VasRevenueLast30d
COMPLAINTS:   ComplaintCountLast90d, DaysSinceLastComplaint, CustCareCallsLast30d
PRODUCT:      ProductCount, HandsetAgeMonths, DataPlanFlag, RoamingFlag
TARGET:       ChurnFlag (1=churned, 0=active)
"""

# ─────────────────────────────────────────────
#  PRE-PROCESSING: Strip SQL aliases
# ─────────────────────────────────────────────

def preprocess_column_names(columns: list) -> dict:
    cleaned = {}
    for col in columns:
        c = col.strip()
        c = re.sub(r'^[a-zA-Z]\.', '', c)
        c = re.sub(r'^[a-zA-Z_]+\.', '', c)
        c = c.strip().lower()
        cleaned[col] = c
    return cleaned

# ─────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────

def field_standardization_agent(state: AgentState) -> dict:
    print("\n" + "="*60)
    print("[Agent 1] Field Standardization — Starting")
    print("="*60)

    raw_columns  = state.get('df_columns', [])
    pre_cleaned  = preprocess_column_names(raw_columns)
    cleaned_cols = list(pre_cleaned.values())

    if cleaned_cols:
        print(f"[Agent 1] Pre-processed {len(raw_columns)} columns. SQL aliases stripped.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=8000
    )

    prompt = f"""You are a Telecom Data Engineering Agent. Rename DataFrame columns to a clean, standardized format.

## INPUTS
SQL QUERY: {state.get('sql_query', 'Not provided')}
COLUMNS (aliases stripped): {cleaned_cols if cleaned_cols else 'see dataset profile'}
DATASET PROFILE: {state.get('metadata_summary', 'Not provided')}
SPECIAL RULES: {state.get('special_rules', 'None')}

## NAMING RULES
1. Use UpperCamelCase (PascalCase). Example: revenueLastMonth → RevenueLast30d
2. Use SHORT but READABLE abbreviations — a non-telecom junior data scientist must understand them.
   GOOD: MouMins, ArpuMonthly, DataUsageKb, ChurnFlag, TenureDays
   BAD: MinutesOfUsageInMinutes (too long), MuMn (unreadable)
3. Keep telecom terms that are industry-standard and widely known: Mou, Arpu, Vas, Msisdn
4. Expand abbreviations that only insiders know: aon→AgeOnNetwork, rchg→Recharge, gndr→Gender
5. Always append unit suffix when relevant: Kb, Mb, Mins, Secs, Days, Months
6. Date integers (YYYYMMDD format): rename to clearly show it's a date e.g. ActivationDate

## ABBREVIATION REFERENCE
{TELECOM_ABBREVIATION_MAP}

## UNIT DETECTION
{TELECOM_DATA_UNIT_RULES}

## DATE DETECTION
{TELECOM_DATE_DETECTION_RULES}

## TARGET SCHEMA (match these when possible)
{STANDARD_CHURN_SCHEMA}

## CONFIDENCE
- CONFIDENT: >85% sure → use the clean name
- AMBIGUOUS: 2+ meanings, unresolvable → prefix new name with ambiguous_
- UNKNOWN: no reasonable interpretation → prefix new name with unknown_

## OUTPUT FORMAT
Return EXACTLY two code blocks, nothing else.

```python
rename_map = {{
    "raw_col": "CleanName",
    "raw_col2": "ambiguous_raw_col2",
    "raw_col3": "unknown_raw_col3"
}}
df = df.rename(columns=rename_map)
```

```python
ambiguous_fields = [
    {{
        "original_column": "raw_col2",
        "candidates": ["OptionA", "OptionB"],
        "reason": "brief reason",
        "sample_values": "actual values from data"
    }}
]
```

RULES:
- No comments, no prose, no extra text outside the two blocks
- Keys must exactly match raw column names
- No duplicate values in rename_map
- Never drop columns
"""

    print("[Agent 1] Sending to LLM...")
    response     = llm.invoke(prompt)

    print("\n--- RAW LLM RESPONSE ---")
    print(response.content)
    print("------------------------\n")

    raw_response = response.content

    # ── Parse code blocks ──
    code_blocks = re.findall(r'```(?:python)?(.*?)```', raw_response, re.DOTALL)
    code_blocks = [block.strip() for block in code_blocks if block.strip()]

    if not code_blocks:
        print("[Agent 1] ERROR: LLM returned no parseable code blocks.")
        return {**state, "cleaning_code": "", "ambiguous_fields": [], "column_map": {}}

    rename_code    = code_blocks[0] if len(code_blocks) >= 1 else ""
    ambiguous_code = code_blocks[1] if len(code_blocks) >= 2 else ""

    # ── Safely extract ambiguous_fields ──
    ambiguous_fields = []
    if ambiguous_code:
        try:
            local_ns = {}
            exec(ambiguous_code, {}, local_ns)
            ambiguous_fields = local_ns.get("ambiguous_fields", [])
        except Exception as e:
            print(f"[Agent 1] WARNING: Could not parse ambiguous_fields: {e}")

    # ── Safely extract rename_map for audit trail ──
    column_map = {}
    if rename_code:
        try:
            map_code = rename_code.split("df = df.rename")[0]
            local_ns = {}
            exec(map_code, {}, local_ns)
            column_map = local_ns.get("rename_map", {})
        except Exception as e:
            print(f"[Agent 1] WARNING: Could not parse rename_map for audit: {e}")

    # ── Surface ambiguous/unknown fields ──
    if ambiguous_fields:
        print(f"\n[Agent 1] ⚠️  {len(ambiguous_fields)} AMBIGUOUS FIELD(S) — Human review needed:\n")
        for field in ambiguous_fields:
            print(f"  Column     : {field.get('original_column')}")
            print(f"  Candidates : {field.get('candidates')}")
            print(f"  Reason     : {field.get('reason')}")
            print(f"  Samples    : {field.get('sample_values')}\n")

    unknown_fields = [k for k, v in column_map.items() if v.startswith("unknown_")]
    if unknown_fields:
        print(f"[Agent 1] ❓ {len(unknown_fields)} UNKNOWN FIELD(S) — Manual mapping required:")
        for f in unknown_fields:
            print(f"  - {f}")

    confident_count = len([v for v in column_map.values() if not (v.startswith("ambiguous_") or v.startswith("unknown_"))])
    print(f"\n[Agent 1] ✅ {confident_count} fields mapped confidently.")
    print(f"[Agent 1] Rename code ready ({len(rename_code)} chars).\n")

    return {
        "cleaning_code"   : rename_code,
        "ambiguous_fields": ambiguous_fields,
        "column_map"      : column_map,
    }
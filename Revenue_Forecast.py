import statistics
import warnings
import pandas as pd
import calendar
import os
import numpy as np
import holidays
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import openpyxl
warnings.filterwarnings('ignore')

us_holidays_list = []
for year in [2023, 2024, 2025, 2026, 2027]:
    us_holidays = holidays.UnitedStates(years=year)
    us_holidays = list(us_holidays.keys())
    us_holidays_list = us_holidays_list + us_holidays
weekday_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

def Read_files(path, revenue_path, customername_mapping_path):
    collection_df = pd.read_csv(path, encoding='latin-1')
    revenue_df = pd.read_csv(revenue_path, encoding='latin-1')
    customername_mapping_df = pd.read_csv(customername_mapping_path, encoding='latin-1')
    return collection_df, revenue_df, customername_mapping_df
def transform_cash_receipt(collection_df):
    collection_df['Doc Date'] = pd.to_datetime(collection_df['Doc Date'], origin='1899-12-30', unit='D')
    collection_df['Collection Date'] = pd.to_datetime(collection_df['Collection Date'], origin='1899-12-30', unit='D')
    collection_df['Due Date'] = pd.to_datetime(collection_df['Due Date'], origin='1899-12-30', unit='D')

    collection_df["Company"] = collection_df["Company"].str.upper()
    collection_df["Company ID"] = collection_df["Company ID"].str.upper()
    collection_df["Cust ID"] = collection_df["Cust ID"].str.upper()
    collection_df["Customer Name"] = collection_df["Customer Name"].str.upper()
    collection_df["BU Leader"] = collection_df["BU Leader"].str.upper()
    collection_df["Collection Agent"] = collection_df["Collection Agent"].str.upper()

    # Get the day from the datetime object
    collection_df['Company'] = collection_df['Company'].replace(np.nan, 'Not Available')
    collection_df['Company ID'] = collection_df['Company ID'].replace(np.nan, 'Not Available')
    collection_df['Cust ID'] = collection_df['Cust ID'].replace(np.nan, 'Not Available')
    collection_df['Customer Name'] = collection_df['Customer Name'].replace(np.nan, 'Not Available')
    collection_df['Doc Day'] = collection_df['Doc Date'].dt.day
    collection_df['Doc Year'] = collection_df['Doc Date'].dt.year
    collection_df['Doc Month'] = collection_df['Doc Date'].dt.month
    collection_df['Doc WeekDay'] = collection_df['Doc Date'].dt.strftime("%A")
    collection_df['Due Day'] = collection_df['Due Date'].dt.day
    collection_df['Due Year'] = collection_df['Due Date'].dt.year
    collection_df['Due Month'] = collection_df['Due Date'].dt.month
    collection_df['Due WeekDay'] = collection_df['Due Date'].dt.strftime("%A")
    collection_df['Collection Day'] = collection_df['Collection Date'].dt.day
    collection_df['Collection Year'] = collection_df['Collection Date'].dt.year
    collection_df['Collection Month'] = collection_df['Collection Date'].dt.month
    collection_df['Collection WeekDay'] = collection_df['Collection Date'].dt.strftime("%A")
    collection_df['set'] = np.where(collection_df['Collection Amount'] > 0, 'Positive', 'Negative')
    collection_df = collection_df[collection_df['set'].isin(['Positive'])]
    collection_df['DueDays'] = collection_df['Due Date'] - collection_df['Doc Date']
    collection_df['DueDays_number'] = [x.days for x in collection_df['DueDays']]
    collection_df['DSO'] = collection_df['Collection Date'] - collection_df['Doc Date']
    collection_df['DSO_number'] = [x.days for x in collection_df['DSO']]
    collection_df['Doc Date'] = pd.to_datetime(collection_df['Doc Date'])
    return collection_df
def frequency_of_maximum_value(numbers):
    # Find the maximum value in the list
    mode_value = statistics.mode(numbers)
    # Count the frequency of the maximum value
    frequency = numbers.count(mode_value)
    return frequency
def probability_weekday_share_data(df):
    df1 = pd.DataFrame(
        df.groupby(['Customer Name', 'Cust ID', 'Collection WeekDay'])['DSO_number'].agg(
            ['count', 'sum', 'mean']).reset_index())
    df2 = pd.DataFrame(
        df.groupby(['Customer Name', 'Cust ID'])['DSO_number'].agg(
            ['count', 'sum', 'mean']).reset_index())
    df1.rename(columns={"count": "weekdaycount"}, inplace=True)

    df3 = pd.merge(df1[['Customer Name', 'Cust ID', 'Collection WeekDay', 'weekdaycount']],
                   df2[['Customer Name', 'Cust ID', 'count']],
                   on=['Customer Name', 'Cust ID'], how="inner")
    df3['weekdayshare'] = df3['weekdaycount'] / df3['count']
    return df3
def predict_date(PDD_date, pred_days):
    pred_days = round(pred_days, 0)
    pred_date_1 = PDD_date + pd.to_timedelta(pred_days, unit='D')
    b = True
    while (b == True):
        pred_weekday_1 = pred_date_1.day_name()
        if pred_weekday_1 == 'Saturday':
            pred_date_2 = pred_date_1 + pd.Timedelta(days=2)
        elif pred_weekday_1 == 'Sunday':
            pred_date_2 = pred_date_1 + pd.Timedelta(days=1)
        else:
            pred_date_2 = pred_date_1
        us_holidays_2023 = holidays.UnitedStates(years=2023)  # Replace with the desired year
        us_holidays_2024 = holidays.UnitedStates(years=2024)
        us_holidays_2023_list = list(us_holidays_2023.keys())
        us_holidays_2024_list = list(us_holidays_2024.keys())
        us_holidays_list = us_holidays_2023_list + us_holidays_2024_list
        # Convert the Timestamp to datetime.date for comparison
        if pred_date_2.date() in us_holidays_list:
            pred_date_1 = pred_date_2 + pd.Timedelta(days=1)
            b = True
        else:
            pred_date_1 = pred_date_2
            b = False
    return pred_date_1, pred_date_1.day_name()
def predict_date_new(pred_date_1):
    us_holidays_2023 = holidays.UnitedStates(years=2023)  # Replace with the desired year
    us_holidays_2024 = holidays.UnitedStates(years=2024)
    us_holidays_2023_list = list(us_holidays_2023.keys())
    us_holidays_2024_list = list(us_holidays_2024.keys())
    us_holidays_list = us_holidays_2023_list + us_holidays_2024_list
    # Convert the Timestamp to datetime.date for comparison
    if pred_date_1.date() in us_holidays_list:
        b = True
    else:
        b = False
    return b
def cal_metrics(collection_df, group, metric, dso_cutoff_date):
    collection_df_dso = pd.DataFrame(collection_df[(collection_df['Data'] == 'past') & (collection_df['Collection Date']>=dso_cutoff_date)].groupby(group)[metric].agg(
        ['mean', 'median', 'std', 'var', 'count', 'min', 'max',
         lambda x: x.mode().iloc[0],
         lambda x: np.percentile(x, 30),
         lambda x: np.percentile(x, 60),
         lambda x: np.percentile(x, 70),
         lambda x: np.percentile(x, 80),
         lambda x: np.percentile(x, 90),
         lambda x: np.percentile(x, 100),
         lambda x: x.quantile(0.75) - x.quantile(0.25),
         lambda x: x.max() - x.min()]).reset_index())

    # Rename columns for clarity
    group.extend(['Mean', 'Median', 'Standard Deviation',
                  'Variance', 'Count', 'Min', 'Max', 'Mode',
                  '30th Percentile', '60th Percentile',
                  '70th Percentile', '80th Percentile',
                  '90th Percentile', '100th Percentile', 'Interquartile Range', 'Range'])
    collection_df_dso.columns = group
    return collection_df_dso
def keep_increasing_values(input_list,days):
    input_list = sorted(input_list)
    output_list = []
    for value in input_list:
        if not output_list or (value > (max(output_list) + days)):
            output_list.append(value)
    return output_list
def keep_increasing_values_new(input_list, percent):
    input_list = sorted(input_list)
    output_list = []
    for value in input_list:
        if not output_list or (value > (max(output_list)*(1+percent))):
            output_list.append(value)
    return output_list
def is_holiday_or_weekend(date, holidays):
    if date.weekday() == 5:  # Saturday
        return 'Saturday'
    elif date.weekday() == 6:  # Sunday
        return 'Sunday'
    elif date.date() in holidays:
        return 'Holiday'
    return None
def adjust_date(pred_date, holidays):
    while True:
        status = is_holiday_or_weekend(pred_date, holidays)
        if status == 'Saturday':
            pred_date += timedelta(days=-1)
        elif status == 'Sunday':
            pred_date += timedelta(days=1)
        elif status == 'Holiday':
            if (is_holiday_or_weekend(pred_date, holidays) == 'Holiday') and (is_holiday_or_weekend(pred_date + timedelta(days=1), holidays) == 'Holiday'):
                if pred_date.weekday() <=2:
                    pred_date += timedelta(days=+2)
                elif pred_date.weekday() == 3:
                    pred_date += timedelta(days=+4)
            elif (is_holiday_or_weekend(pred_date, holidays) == 'Holiday') and (is_holiday_or_weekend(pred_date - timedelta(days=1), holidays) == 'Holiday'):
                if pred_date.weekday() == 4:
                    pred_date += timedelta(days=+3)
                else:
                    pred_date += timedelta(days=+1)
            elif (is_holiday_or_weekend(pred_date, holidays) == 'Holiday'):
                if pred_date.weekday() == 4:
                    pred_date += timedelta(days=-1)
                else:
                    pred_date += timedelta(days=+1)
        else:
            break
    return pred_date, pred_date.day_name()
def get_week(temp, p, pred_date_1):
    cust_name = p['Customer Name']
    cust_id = p['Cust ID']
    temp1 = temp[(temp['Customer Name'] == cust_name) & (temp['Cust ID'] == cust_id)]
    temp1.sort_values(by='weekdayshare', ascending=False, inplace=True)
    if temp1['Collection WeekDay'].to_list() == []:
        newdt, newweekday = adjust_date(pred_date_1, us_holidays_list)
    else:
        for k in range(len(temp1['Collection WeekDay'].to_list())):
            weekday1 = temp1['Collection WeekDay'].to_list()[k]
            weekday2 = pred_date_1.day_name()
            # Calculate the difference in days
            day_difference = (weekday_mapping[weekday1] - weekday_mapping[weekday2] + 7) % 7
            pred_date_1 = pred_date_1 + pd.Timedelta(days=day_difference)
            newdt, newweekday = adjust_date(pred_date_1, us_holidays_list)
            if newdt == pred_date_1:
                break
            else:
                pred_date_1 = newdt
    d1 = newdt
    return d1
def create_group_metrics(collection_df, dso_cutoff_date):
    df_company = cal_metrics(collection_df, ['Company'], 'DSO (days)', dso_cutoff_date)
    df_companyid = cal_metrics(collection_df, ['Company ID'], 'DSO (days)', dso_cutoff_date)
    df_customer = cal_metrics(collection_df, ['Customer Name'], 'DSO (days)', dso_cutoff_date)
    df_customer_custid = cal_metrics(collection_df, ['Customer Name', 'Cust ID'], 'DSO (days)', dso_cutoff_date)
    df_custid = cal_metrics(collection_df, ['Company', 'Cust ID'], 'DSO (days)', dso_cutoff_date)
    df_company.to_csv(os.path.join(path, "Output", "df_company.csv"))
    df_companyid.to_csv(os.path.join(path, "Output", "df_companyid.csv"))
    df_customer.to_csv(os.path.join(path, "Output", "df_customer.csv"))
    df_customer_custid.to_csv(os.path.join(path, "Output", "df_customer_custid.csv"))
    df_custid.to_csv(os.path.join(path, "Output", "df_custid.csv"))
    return df_company, df_companyid, df_customer, df_customer_custid, df_custid
def predict_AR(collection_df, probability_weekday_df, cutoff_date, dso_cutoff_date):
    df_company, df_companyid, df_customer, df_customer_custid, df_custid = create_group_metrics(collection_df, dso_cutoff_date)
    present = collection_df[collection_df['Data'] == 'present']
    present.reset_index(inplace=True)
    present['aging'] = cutoff_date - present['Doc Date']
    present['aging'] = [x.days for x in present['aging']]
    present = present.loc[present['aging'] <= 120, :]
    present.reset_index(inplace=True)
    for index in range(present.shape[0]):
        a = df_custid[(df_custid['Cust ID'] == present.loc[index]['Cust ID']) & (df_custid['Company'] == present.loc[index]['Company'])]
        # a = df_customer[(df_customer['Customer Name'] == present.loc[index]['Customer Name'])]
        b = df_companyid[(df_companyid['Company ID'] == present.loc[index]['Company ID'])]
        c = df_company[(df_company['Company'] == present.loc[index]['Company'])]
        k = []
        if a.shape[0] > 0:
            k = k + \
                [a["Median"][a.index[0]],
                 a["Mode"][a.index[0]],
                 a["Mean"][a.index[0]]
                 ]
            k = k + \
                [
                 a["80th Percentile"][a.index[0]],
                 a["100th Percentile"][a.index[0]]
                    # ,a["100th Percentile"][a.index[0]] + 15,
                    # a["100th Percentile"][a.index[0]] + 30,
                    # a["100th Percentile"][a.index[0]] + 45,
                    # a["100th Percentile"][a.index[0]] + 60,
                    # a["100th Percentile"][a.index[0]] + 75,
                    # a["100th Percentile"][a.index[0]] + 90,
                    # a["100th Percentile"][a.index[0]] + 105,
                    # a["100th Percentile"][a.index[0]] + 120,
                    # a["100th Percentile"][a.index[0]] + 135
                 ]
        m = keep_increasing_values(k, 14)
        p = present.iloc[index, :]
        for i in m:
            pred_days = i - 4
            pred_days = round(pred_days, 0)
            pred_date_1 = p['Doc Date'] + pd.to_timedelta(pred_days, unit='D')
            d1 = get_week(probability_weekday_df, p, pred_date_1)
            diff = d1 - cutoff_date
            if diff.days > 0:
                present.loc[index, "Date of Prediction"] = d1
                break
        # for i in m:
        #     pred_days = i
        #     pred_days = round(pred_days, 0)
        #     pred_date_1 = p['Doc Date'] + pd.to_timedelta(pred_days, unit='D')
        #     if ((pred_date_1-cutoff_date).days >-7) and ((pred_date_1-cutoff_date).days < 10):
        #         pred_date_1 = pred_date_1 + pd.to_timedelta(2 - (pred_date_1-cutoff_date).days, unit='D')
        #     elif ((pred_date_1-cutoff_date).days >= 10):
        #         pred_date_1 = pred_date_1 - pd.to_timedelta(7, unit='D')
        #     d1 = get_week(probability_weekday_df, p, pred_date_1)
        #     diff = d1 - cutoff_date
        #     if diff.days > 0:
        #         present.loc[index, "Date of Prediction"] = d1
        #         break
    return present
def format_predict_AR(present):
    present['Week_prediction'] = present['Date of Prediction'] - cutoff_date
    present['Week_prediction'] = [x.days for x in present['Week_prediction']]
    present['Week_model'] = [(x - 1) // 7 + 1 for x in present['Week_prediction']]
    present['PredictionWeek'] = present['Date of Prediction'].dt.strftime("%A")
    # pd.pivot_table(present,
    #                values='Collection Amount',
    #                index='Week_model',
    #                columns='PredictionWeek', aggfunc=sum)
    return present
def predict_AR_duedate(present, probability_weekday_df, cutoff_date):
    present["Due Date Prediction"] = pd.NaT
    for index in range(present.shape[0]):
        # print(index)
        p = present.iloc[index, :]
        d1 = get_week(probability_weekday_df, p, p["Due Date"])
        for i in range(0):
            if (d1 + pd.to_timedelta(i * 14, unit='D')) > cutoff_date:
                present.loc[index, "Due Date Prediction"] = d1 + pd.to_timedelta(i * 14, unit='D')
                break
    return present
def final_format_predict_AR(present):
    # present[["Date of Prediction",'Due Date', 'Due Date new']]
    present['Date of Prediction1'] = [pd.isna(x) for x in present['Date of Prediction']]
    present['Date of Prediction final'] = [x if pd.isna(y) else y for x, y in
                                           zip(present['Due Date Prediction'], present['Date of Prediction'])]
    # present['Date of Prediction final1'] = [x + pd.DateOffset(days=7 - x.weekday()) if x.weekday() in [5, 6] else x for x in present['Date of Prediction final']]

    present['Week_prediction'] = present['Date of Prediction final'] - cutoff_date
    present['Week_prediction'] = [x.days for x in present['Week_prediction']]
    present['Week_model'] = [(x - 1) // 7 + 1 for x in present['Week_prediction']]
    return present
def Output_dashboard_ARAging(present):
    Output_dashboard = present[["Company", "Company ID", "Cust ID",
     "Customer Name", "BU Leader", "Collection Agent", "Ref Nbr", "Doc Date",
     "Collection Amount", "Date of Prediction final"]]
    Output_dashboard.rename(columns={'Date of Prediction final': 'Date of Prediction'}, inplace=True)
    Output_dashboard['Type'] = "ARAging"
    Output_dashboard = Output_dashboard[Output_dashboard['Date of Prediction'].notna()]
    return Output_dashboard
def format_file(path, revenue_path, customername_mapping_path):
    collection_df, revenue_df, customername_mapping_df = Read_files(path, revenue_path, customername_mapping_path)
    collection_df = transform_cash_receipt(collection_df)
    past_df = collection_df[collection_df['Data'] == "past"]
    probability_weekday_df = probability_weekday_share_data(past_df)
    revenue_df = tranform_revenue_df(revenue_df)
    customername_mapping_df = tranform_customername_mapping_df(customername_mapping_df)
    return collection_df, revenue_df, customername_mapping_df, probability_weekday_df
def run_araging(collection_df, probability_weekday_df, cutoff_date, dso_cutoff_date):
    present = predict_AR(collection_df, probability_weekday_df, cutoff_date, dso_cutoff_date)
    present = format_predict_AR(present)
    present = predict_AR_duedate(present, probability_weekday_df, cutoff_date)
    present = final_format_predict_AR(present)
    Output_dashboard = Output_dashboard_ARAging(present)
    tm = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    present_pivot = present.groupby(['Week_model'])['Collection Amount'].sum()
    present_pivot.to_csv(os.path.join(path, "Output", "AR_Aging_pivot" + tm + ".csv"))
    present.to_csv(os.path.join(path, "Output", "AR_Aging_Output" + tm + ".csv"))
    probability_weekday_df.to_csv(os.path.join(path, "temp.csv"))
    Output_dashboard.to_excel(os.path.join(path, "Output", "Output_dashboard.xlsx"), sheet_name="Dashboard", engine='openpyxl', index=False)
    return present, Output_dashboard, present_pivot
def tranform_revenue_df(revenue_df):
    revenue_df["Customer Name1"] = revenue_df["Customer Name1"].str.upper()
    revenue_df["Unique ID"] = revenue_df["Unique ID"].astype(str).str.upper()
    revenue_df["Billing Customer ID"] = revenue_df["Billing Customer ID"].astype(str).str.upper()
    return revenue_df
def tranform_customername_mapping_df(customername_mapping_df):
    customername_mapping_df.rename(columns={"Ext. Ref.": "Cust ID"}, inplace=True)
    customername_mapping_df["Billing Customer ID"] = customername_mapping_df["Billing Customer ID"].astype(str).str.upper()
    customername_mapping_df["Unique ID"] = customername_mapping_df["Unique ID"].astype(str).str.upper()
    customername_mapping_df["Customer Name"] = customername_mapping_df["Customer Name"].astype(str).str.upper()
    customername_mapping_df["Customer Name1"] = customername_mapping_df["Customer Name1"].astype(str).str.upper()
    customername_mapping_df["Cust ID"] = customername_mapping_df["Cust ID"].astype(str).str.upper()
    return customername_mapping_df
def adjust_day(day, year, month):
    # Function to adjust the day based on the target month and year
    last_day_of_month = calendar.monthrange(year, month)[1]
    return min(day, last_day_of_month)
def mean_last_3_months(group):
    # Define a function to calculate the mean sales for the last three months
    # Find the maximum date in the 'date1' column
    max_date = group['Doc Date'].max()

    # Calculate the date 90 days before the maximum date
    start_date = max_date - pd.Timedelta(days=90)

    # Filter the DataFrame to include only the rows up to 90 days from the maximum date
    last_3_months = group[(group['Doc Date'] >= start_date) & (group['Doc Date'] <= max_date)]

    # Calculate the mean of the sales for the last three months
    return last_3_months['DSO (days)'].mean()
def freq_day_last_3_months(group):
    # Define a function to calculate the mean sales for the last three months
    # Find the maximum date in the 'date1' column
    max_date = group['Doc Date'].max()

    # Calculate the date 90 days before the maximum date
    start_date = max_date - pd.Timedelta(days=90)

    # Filter the DataFrame to include only the rows up to 90 days from the maximum date
    last_3_months = group[(group['Doc Date'] >= start_date) & (group['Doc Date'] <= max_date)]
    last_3_months['Doc Date Day'] = last_3_months['Doc Date'].dt.day
    day = last_3_months['Doc Date Day'].mean()
    # Calculate the mean of the sales for the last three months
    return day
def run_forecast(revenue_df, customername_mapping_df, cutoff_date, collection_df, probability_weekday_df):
    first_day_of_current_month = cutoff_date.replace(day=1)
    previous_month_last_date = first_day_of_current_month - pd.DateOffset(days=1)
    filtered_df1 = collection_df[(collection_df['Data'] == "past")]
    filtered_df1 = filtered_df1.sort_values(by=['Cust ID', 'Doc Date'], ascending=[True, False])
    grouped = filtered_df1.groupby(['Cust ID'])
    result = grouped.apply(mean_last_3_months).reset_index(name='mean_last_3_months')
    result["Cust ID"] = result["Cust ID"].str.upper()

    filtered_df1 = collection_df[(collection_df['Doc Date'] <= previous_month_last_date)]
    # Sort the DataFrame by group and date in descending order
    filtered_df1 = filtered_df1.sort_values(by=['Cust ID', 'Doc Date'], ascending=[True, False])
    # Group the DataFrame by 'group' column
    grouped = filtered_df1.groupby(['Cust ID'])

    # Apply the function to each group and reset index
    result_freqday = grouped.apply(freq_day_last_3_months).reset_index(name='freq_day_last_3_months')
    result_freqday["Cust ID"] = result_freqday["Cust ID"].str.upper()

    df3 = pd.merge(revenue_df, customername_mapping_df, on=['BFC', 'SFC', 'Billing Customer ID', 'Unique ID', 'Customer Name1'], how="left")
    df3["Cust ID"] = df3["Cust ID"].str.upper()

    df4 = pd.merge(df3, result, on=['Cust ID'], how="left")
    df4['mean_last_3_months'] = df4['mean_last_3_months'].fillna(30)

    df5 = pd.merge(df4, result_freqday, on=['Cust ID'], how="left")
    df5['freq_day_last_3_months'] = df5['freq_day_last_3_months'].fillna(30)

    # Extract the day of the month
    target_year = cutoff_date.year
    target_month = cutoff_date.month
    df5['Adjusted Day'] = df5['freq_day_last_3_months'].apply(adjust_day, args=(target_year, target_month))
    df5['first pred invoice date'] = pd.to_datetime({'year': target_year, 'month': target_month, 'day': df5['Adjusted Day']})

    for i in [0, 1, 2, 3]:
        df5['invoice date_' + str(target_month + i) + '_' + str(target_year)] = [x + relativedelta(months=i) if not(pd.isna(x)) else "" for x in df5['first pred invoice date']]
        df5['pred date_' + str(target_month + i) + '_' + str(target_year)] = [x + pd.to_timedelta(y, unit='D') if x!="" and not pd.isna(y) else "" for x,y in zip(df5['invoice date_' + str(target_month + i) + '_' + str(target_year)], df5['mean_last_3_months'])]
        df5['pred date_' + str(target_month + i) + '_' + str(target_year) + "_1"] = [x + pd.DateOffset(days=7 - x.weekday()) if x.weekday() in [5, 6] else x for x in df5['pred date_' + str(target_month + i) + '_' + str(target_year)]]
    for i in [0, 1, 2, 3]:
        for index in range(df5.shape[0]):
            p = df5.iloc[index, :]
            pred_date_1 = p['pred date_' + str(target_month + i) + '_' + str(target_year) + "_1"]
            d1 = get_week(probability_weekday_df, p, pred_date_1)
            df5.loc[index, 'pred date_' + str(target_month + i) + '_' + str(target_year) + "_1"] = d1

    df5 = df5[~df5['Cust ID'].str.contains('TBD', na=False)]
    # df5 = df5[~df5['Cust ID'].str.contains('NAN', na=False)]
    # df5 = df5[~df5['Cust ID'].str.contains('NaN', na=False)]
    # df5 = df5[~df5['Cust ID'].str.contains('0', na=False)]
    # # df5 = df5.dropna(subset=['Cust ID'])

    pdf = pd.DataFrame()
    for i, j in zip([0, 1, 2, 3], ["Jul'24 Fcst", "Aug'24 Fcst", "Sep'24 Fcst", "Oct'24 Fcst"]):
        df = df5[["invoice date_" + str(target_month + i) + "_" + str(target_year), j, "pred date_" + str(target_month + i) + '_' + str(target_year) + "_1", 'Customer Name', 'Cust ID', 'BFC', 'SFC', 'Billing Customer ID', 'Unique ID']]
        df.rename(columns={"invoice date_" + str(target_month + i) + "_" + str(target_year): "Inv Date"}, inplace=True)
        df.rename(columns={j: "Amount"}, inplace=True)
        df.rename(columns={'pred date_' + str(target_month + i) + "_" + str(target_year) + "_1": "Pred Date"}, inplace=True)
        pdf = pd.concat([pdf, df])

    # pdf = pdf[(pdf['Inv Date'] > max(collection_df['Doc Date']))]
    pdf = pdf[(pdf['Inv Date'] > cutoff_date)]
    pdf = pdf[(pdf['Amount'] > 0)]

    pdf['Week_prediction'] = pdf['Pred Date'] - cutoff_date
    pdf['Week_prediction'] = [x.days for x in pdf['Week_prediction']]
    pdf['Week_model'] = [(x - 1) // 7 + 1 for x in pdf['Week_prediction']]
    pdf_weekmodel = pdf.groupby(['Week_model'])['Amount'].sum()
    pdf_weekmodel = pd.DataFrame(pdf_weekmodel)
    pdf_weekmodel.reset_index(inplace=True)
    index = pdf[['Pred Date', 'Week_model']]
    index = index.groupby(['Week_model'])['Pred Date'].min()
    index = pd.DataFrame(index)
    index.reset_index(inplace=True)
    index['Pred Date'] = index['Pred Date'].dt.date
    pdf_weekmodel = pd.merge(pdf_weekmodel, index, on=['Week_model'], how="inner")
    tm = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    result.to_csv(os.path.join(path, "Forecast", "mean_DSO_last_3_months" + tm + ".csv"))
    result_freqday.to_csv(os.path.join(path, "Forecast", "freq_day_last_3_months" + tm + ".csv"))
    df5.to_csv(os.path.join(path, "Forecast", "Forecast_complete_file" + tm + ".csv"))
    pdf_weekmodel.to_csv(os.path.join(path, "Forecast", "Forecast_pivot" + tm + ".csv"))
    pdf.to_csv(os.path.join(path, "Forecast", "Forecast_records" + tm + ".csv"))
    forecast_df = pdf
    return pdf_weekmodel, forecast_df


cutoff_date = pd.Timestamp('2024-07-28')
dso_cutoff_date = pd.Timestamp('2024-01-01')
path = r"D:\CLient\Work\28072024"
cash_receipt_file = os.path.join(path, "ARAging_28_07_2024_new.csv")
revenue_path = r"D:\CLient\Revenue Forecast\Americas Obligor & Non-Obligor Revenue 24th Apr'24.csv"
customername_mapping_path = r"D:\CLient\Revenue Forecast\bfc_sfc_customerid_mapping_new_24_06_2024_both.csv"
custid_mapping_path = r"D:\CLient\Revenue Forecast\Custid_Company_BU_Mapping.csv"
collection_df, revenue_df, customername_mapping_df, probability_weekday_df = format_file(cash_receipt_file, revenue_path, customername_mapping_path)
present, Output_dashboard, present_pivot = run_araging(collection_df, probability_weekday_df, cutoff_date, dso_cutoff_date)
forecast_pivot, forecast_df = run_forecast(revenue_df, customername_mapping_df, cutoff_date, collection_df, probability_weekday_df)
print("Forecasting Completed Successfully!")

#
# def forecast_dashboard_ARAging(forecast_df, custid_mapping_path):
#     forecast_df.rename(columns={'Inv Date': 'Doc Date',
#     'Pred Date': 'Date of Prediction',
#     'Amount': 'Collection Amount'}, inplace=True)
#     forecast_df["Ref Nbr"] = ""
#     custid_mapping = pd.read_csv(custid_mapping_path, encoding='latin-1')
#     forecast_df = pd.merge(forecast_df, custid_mapping, on=['Customer Name', 'Cust ID'], how="left")
#     forecast_df['Type'] = "Forecast"
#     forecast_df['Date of Prediction'] = forecast_df['Date of Prediction'].dt.date
#     forecast_df_Dashboard = forecast_df[['Company', 'Company ID', 'Cust ID', 'Customer Name', 'BU Leader',
#            'Collection Agent', 'Ref Nbr', 'Doc Date', 'Collection Amount',
#            'Date of Prediction', 'Type']]
#     return forecast_df_Dashboard
#
#
# def forecast_dashboard_ARAging(forecast_df, custid_mapping_path):
#     forecast_df.rename(columns={'Inv Date': 'Doc Date',
#     'Pred Date': 'Date of Prediction',
#     'Amount': 'Collection Amount'}, inplace=True)
#     forecast_df["Ref Nbr"] = ""
#     # custid_mapping = pd.read_csv(custid_mapping_path, encoding='latin-1')
#     # forecast_df = pd.merge(forecast_df, custid_mapping, on=['Customer Name', 'Cust ID'], how="left")
#     forecast_df['Type'] = "Forecast"
#     forecast_df['Company'] = ""
#     forecast_df['Company ID'] = ""
#     forecast_df['BU Leader'] = ""
#     forecast_df['Collection Agent'] = ""
#     forecast_df['Date of Prediction'] = forecast_df['Date of Prediction'].dt.date
#     forecast_df_Dashboard = forecast_df[['Company', 'Company ID', 'Cust ID', 'Customer Name', 'BU Leader',
#            'Collection Agent', 'Ref Nbr', 'Doc Date', 'Collection Amount',
#            'Date of Prediction', 'Type']]
#     return forecast_df_Dashboard
#
# forecast_df_Dashboard = forecast_dashboard_ARAging(forecast_df, custid_mapping_path)
# forecast_df_Dashboard.to_excel(os.path.join(path, "Forecast", "Forecast_dashboard.xlsx"),
#                                sheet_name="Dashboard",
#                                engine='openpyxl',
#                                index=False)
#
# pd.concat([Output_dashboard, forecast_df_Dashboard]).to_excel(os.path.join(path, "Final_dashboard.xlsx"),
#                                                               sheet_name="Dashboard",
#                                                               engine='openpyxl',
#                                                               index=False)
#

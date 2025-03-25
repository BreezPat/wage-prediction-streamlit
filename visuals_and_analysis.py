import streamlit as st
import pandas as pd
import plotly.express as px

education_levels = ['less_than_hs', 'high_school', 'some_college', 'bachelors_degree', 'advanced_degree']


def display_dataset(wages_dif):
    st.image("stream_banner.webp", use_column_width=True)
    st.title('Wages by Education in the USA (1973-2022)')
    st.write('This app displays the dataset on wages by education in the USA.')
    st.write('## About the Dataset')
    st.write('This dataset contains information about average hourly wages for different education levels in the USA from 1973 to 2022. It includes breakdowns by gender and race/ethnicity.')
    st.markdown("<h2 style='color: orange;'>Data Overview</h2>", unsafe_allow_html=True)
    st.write(wages_dif.head())




def display_statistics(wages_dif):
    st.markdown("<h2 style='color: orange;'>Basic Statistics</h2>", unsafe_allow_html=True)
    st.write(wages_dif.describe())
    st.markdown("<h2 style='color: orange;'>Data Types</h2>", unsafe_allow_html=True)
    data_types = pd.DataFrame(wages_dif.dtypes.astype(str), columns=['Data Type']).reset_index()
    data_types.rename(columns={'index': 'Column'}, inplace=True)
    st.dataframe(data_types)
    st.markdown("<h2 style='color: orange;'>Missing Values</h2>", unsafe_allow_html=True)
    st.write(wages_dif.isnull().sum())




def display_time_series(wages_dif):
    st.markdown("<h2 style='color: orange;'>Average Wages Over the Years</h2>", unsafe_allow_html=True)
    fig = px.line(wages_dif, x='year', y=['less_than_hs', 'high_school', 'some_college', 'bachelors_degree', 'advanced_degree'],
                  title='***Time Series: Distribution of Average Wages Over the Years by Education Level',
                  labels={'Wages': 'Wages ($)'})
    st.plotly_chart(fig)




def display_box_plot(wages_dif):
    reshaped_df = pd.melt(wages_dif, id_vars=['year'], value_vars=['high_school', 'bachelors_degree', 'advanced_degree'],
                          var_name='Education Level', value_name='Wages')
    
    # Define a custom neon color palette
    neon_colors = ['#39ff14', 'cyan', '#ff073a']

    box_fig = px.box(reshaped_df, x='Education Level', y='Wages', color='Education Level',
                     title='***Box Plot: Distribution of Wages by Education Level',
                     labels={'Wages': 'Wages ($)'},
                     color_discrete_map={key: color for key, color in zip(reshaped_df['Education Level'].unique(), neon_colors)})

    # Update the layout to increase the size
    box_fig.update_layout(height=800, width=1000)
    
    st.plotly_chart(box_fig)



def average_hourly_wages_by_education(wages_dif):
    education_levels = ['less_than_hs', 'high_school', 'some_college', 'bachelors_degree', 'advanced_degree']
    st.markdown("<h4 style='color: lime;'>Select Education Level:</h4>", unsafe_allow_html=True)
    selected_education = st.selectbox('Select Education Level:', education_levels, label_visibility='collapsed')
    filtered_data = wages_dif[['year', selected_education]]
    st.markdown(f"<h2 style='color: orange;'>Average Wages Over the Years for {selected_education.replace('_', ' ').title()}</h2>", unsafe_allow_html=True)
    fig = px.line(filtered_data, x='year', y=selected_education,
                  title=f'*** Average Wages Over the Years for {selected_education.replace("_", " ").title()}',
                  color_discrete_sequence=['lime'])
    fig.update_layout(xaxis_title='Year', yaxis_title='Average Wages',
                      xaxis=dict(title_font=dict(size=16, color='white')),
                      yaxis=dict(title_font=dict(size=16, color='white')))
    fig.update_traces(line=dict(width=4))
    st.plotly_chart(fig)




def hourly_wages_by_gender(wages_dif):
    st.markdown("<h2 style='color: orange;'>Hourly Wages by Gender in the US</h2>", unsafe_allow_html=True)
    st.markdown("<h6>***Wages based on education level in the USA: Men vs Women (1973-2022)</h6>", unsafe_allow_html=True)
    men_col = ['men_less_than_hs', 'men_high_school', 'men_some_college', 'men_bachelors_degree', 'men_advanced_degree']
    women_col = ['women_less_than_hs', 'women_high_school', 'women_some_college', 'women_bachelors_degree', 'women_advanced_degree']
    men_wages = wages_dif[men_col].mean()
    women_wages = wages_dif[women_col].mean()
    wages_diff = pd.DataFrame({'Education Level': education_levels, 'Men': men_wages.values, 'Women': women_wages.values})
    bar_fig = px.bar(wages_diff, x='Education Level', y=['Men', 'Women'], 
                     barmode='group', 
                     labels={'Wages': 'Wages ($)'},
                     color_discrete_map={'Men': 'blue', 'Women': 'orange'})
    st.plotly_chart(bar_fig)




def hourly_wages_by_gender_and_education_level(wages_dif):
    st.markdown("<h4 style='color: lime;'>Select Education Level:</h4>", unsafe_allow_html=True)
    education_level = st.radio("Education Level", ('High School', 'Bachelor\'s Degree', 'Advanced Degree'), label_visibility='collapsed')

    if education_level == 'High School':
        men_col = 'men_high_school'
        women_col = 'women_high_school'
    elif education_level == 'Bachelor\'s Degree':
        men_col = 'men_bachelors_degree'
        women_col = 'women_bachelors_degree'
    else:
        men_col = 'men_advanced_degree'
        women_col = 'women_advanced_degree'

    st.markdown(f"<h3 style='color: orange;'>***Bar Chart: Hourly Wage after a {education_level} Based on Gender</h3>", unsafe_allow_html=True)

    selected_years = [1982, 1992, 2002, 2012, 2022]
    df_selected_years = wages_dif[wages_dif['year'].isin(selected_years)]

    fig = px.bar(df_selected_years, x='year', y=[men_col, women_col], barmode='group',
                 labels={'value': 'Hourly Wage ($)', 'variable': 'Gender'},
                 color_discrete_map={men_col: '#1f77b4', women_col: '#ff7f0e'})

    fig.update_layout(width=700, height=650, xaxis_title='Year', yaxis_title='Hourly Wage ($)', legend_title='Gender',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      xaxis=dict(title_font=dict(size=16, color='white')),
                      yaxis=dict(title_font=dict(size=16, color='white')))

    st.plotly_chart(fig)





def hourly_wages_by_race_and_education_level(wages_dif):
    st.markdown("<h4 style='color: lime;'>Select Education Level for Race Analysis:</h4>", unsafe_allow_html=True)
    education_level_race = st.radio("Select Education Level for Race Analysis:",
                                    ('High School', 'Bachelor\'s Degree', 'Advanced Degree'),
                                    label_visibility='collapsed')

    if education_level_race == 'High School':
        race_cols = ['black_high_school', 'hispanic_high_school', 'white_high_school']
        chart_title_race = '*** Hourly Wage after a High School Degree Based on Race'
    elif education_level_race == 'Bachelor\'s Degree':
        race_cols = ['black_bachelors_degree', 'hispanic_bachelors_degree', 'white_bachelors_degree']
        chart_title_race = '*** Hourly Wage after a Bachelor Degree Based on Race'
    else:
        race_cols = ['black_advanced_degree', 'hispanic_advanced_degree', 'white_advanced_degree']
        chart_title_race = '*** Hourly Wage after an Advanced Degree Based on Race'

    selected_years = [1982, 1992, 2002, 2012, 2022]
    df_selected_years_race = wages_dif[wages_dif['year'].isin(selected_years)]

    fig_race = px.bar(df_selected_years_race, x='year', y=race_cols, barmode='group', title=chart_title_race,
                      labels={'value': 'Hourly Wage ($)', 'variable': 'Race'},
                      color_discrete_map={'black_high_school': 'blue', 'hispanic_high_school': 'orange', 'white_high_school': 'green',
                                          'black_bachelors_degree': 'blue', 'hispanic_bachelors_degree': 'orange', 'white_bachelors_degree': 'green',
                                          'black_advanced_degree': 'blue', 'hispanic_advanced_degree': 'orange', 'white_advanced_degree': 'green'})

    fig_race.update_layout(width=700, height=600,
                           xaxis=dict(title_font=dict(size=16, color='white')),
                           yaxis=dict(title_font=dict(size=16, color='white')))

    st.plotly_chart(fig_race)

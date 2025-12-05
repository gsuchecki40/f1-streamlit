## Formula One Streamlit Model

[http://www.f1pred.streamlit.app/ ](https://f1pred.streamlit.app/)

This model operates with the Streamlit and a playwright scraper file that can be easily modified to predict any Grand Prix on the schedule

## Scraper

Inside of the Qualyfing_Scrape.py code, all that would need to be changed to change races is the country name code that the formula1 website uses on their html inside of their dropdown lists. Example Below:

```python
page.get_by_label("All").get_by_role(
            "link", name="COUNTRY_HTML_TAG_HERE"
```

This scraper generates a csv of the qualifying results from the race that is downloaded to your desired destination


## Running the Streamlit Tips

*NOTICE: The Streamlit Cloud does take a few minutes to load the application in*

A little bit of math is required (or a quick ai search) for the points proportion.

Finding the weather information for each race is pretty simple as long as you are specific in what you are asking for.

## How the Model Works

The model used is an XGBoost ML model that takes in the following variables to predict race results.

**Variables**
- Weather Info: Located on the sliders and text inputs
- AvgQualiTime: Takes the average of all Q1, Q2, and Q3 to factor into the prediciton
- PointsProp: The proportion of points that a driver has earned throughout the season compared to what is available

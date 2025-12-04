Link to Streamlit App

http://www.f1pred.streamlit.app/ 

## Formula One Streamlit Model

This model operates with the Streamlit and a playwright scraper file that can be easily modified to predict any Grand Prix on the schedule

## Scraper

Inside of the Qualyfing_Scrape code, all that would need to be changed for the user is the country name for when the scraper chooses a race to pull results from. Example Below:

```python
#page.get_by_label("All").get_by_role(
            #"link", name="COUNTRY_HTML_TAG_HERE"
```

This scraper generates a csv of the qualifying results from the race that is downloaded to your desired destination


## Running the Streamlit

*NOTICE: The Streamlit Cloud does take a few minutes to load the application in*

A little bit of math is required (or a quick ai search) of the points proportion.

Finding the weather information for each race is pretty simple as long as you are specific in what you are asking for.

## How the Model Works

The model used is an XGBoost ML model that takes in the following variables to predict race results.

- Weather Info: Located on the sliders and text inputs
- AvgQualiTime: Takes the average of all Q1, Q2, and Q3 to factor into the prediciton
- PointsProp: The proportion of points that a driver has earned throughout the season compared to what is available

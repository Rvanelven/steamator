# STEAMATOR

## Project Description

A good descriotion of the project...

## Documentation

### Datasets

Liens vers les datasets originaucx de Steam Spy :

**> 27 033 Unique values**

[**steam.csv**](https://www.kaggle.com/nikdavis/steam-store-games?select=steam.csv) \
[**steam_description_data.csv**](https://www.kaggle.com/nikdavis/steam-store-games?select=steam_description_data.csv) \
[**steam_media_data.csv**](https://www.kaggle.com/nikdavis/steam-store-games?select=steam_media_data.csv) \
[**steam_requirements_data.csv**](https://www.kaggle.com/nikdavis/steam-store-games?select=steam_requirements_data.csv) \
[**steam_support_info.csv**](https://www.kaggle.com/nikdavis/steam-store-games?select=steam_support_info.csv) \
[**steamspy_tag_data.csv**](https://www.kaggle.com/nikdavis/steam-store-games?select=steamspy_tag_data.csv)


### Dataset FINAL devra ressembler au tableau ci-dessus (voir PREVIEW MODE) ↓


⚠️ ⚠️ ⚠️ Ne pas modifier les lignes en dessous


|  **column_name** |  steam_appid |  name  |  release_date |  english | developer  |  publisher |  platforms | required_age  |  categories |  genres | steamspy_tags  | achievements  | positive_ratings  | negative_ratings  | average_playtime  | median_playtime  | owners  | price  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  **format_data**  |  id  |  name  |  date  |  language  |  name  |  name  |  os  |  age  |  game categories  |  genres  |  game tag  |  numeric  |  numeric  | numeric  |  minutes  |  minutes  |  numeric  |  price  |
|  **data_types**  |  int64  |  object  |  object  |  int64  |  object  |  object  |  object  |  int64  |  object  |  object  |  object  |  int64  |  int64  |  int64  |  int64  |  int64  |  object  |  float64  |
|  **data_description**  |  Unique identifier for each title  |  Title of app (game)  |  Release date in format YYYY-MM-DD  |  Language support: 1 if is in English  |  Name (or names) of developer(s)  |  Name (or names) of publisher(s)  |  Semicolon delimited list of supported platforms  |  Minimum required age according to PEGI UK standards  |  Semicolon delimited list of game categories, e.g. single-player;multi-player  |  Semicolon delimited list of game genres, e.g. action;adventure  |  Semicolon delimited list of top steamspy game tags, similar to genres but community voted, e.g. action;adventure  |  Number of in-games achievements, if any  |  Number of positive ratings, from SteamSpy  |  Number of negative ratings, from SteamSpy  |  Average user playtime, from SteamSpy  |  Median user playtime, from SteamSpy  |  Estimated number of owners. Contains lower and upper bound (like 20000-50000). May wish to take mid-point or lower  |  Current full price of title in GBP, (pounds sterling)  |


⚠️ ⚠️ ⚠️ Ne pas modifier les lignes au-dessus ⚠️ ⚠️ ⚠️ 



- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for steamator in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/steamator`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "steamator"
git remote add origin git@github.com:{group}/steamator.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
steamator-run
```

# Install

Go to `https://github.com/{group}/steamator` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/steamator.git
cd steamator
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
steamator-run
```

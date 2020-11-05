# 3_body_problem_bot

## How to deploy this bot using Docker:

1. Clone this repository:
   ```
   git clone https://github.com/robolamp/3_body_problem_bot.git
   ```
2. Enter the directry with cloned repo:
   ```
   cd 3_body_problem_bot
   ```

3. Edit `Dockerflie`: replace `[TOKEN]` and `[BOT_NAME]` with your bot token and your channel name

   ```
   RUN (crontab -l ; echo "0 */12 * * * python3 /home/3_body_problem_bot/generate_3_body_simulation.py -V -T [TOKEN] -N [BOT_NAME] --fps 15 --min-score 24 --duration 150 >> /var/log/cron.log") | crontab
   ```

4. Build docker image using command:
   ```
   docker build -t [your image name] -f ./Dockerfile .
   ```
5. Using built image, run the container in background:
   ```
   docker run -d [your image name]:latest
   ```
6. Check that the container is running and find id of your container:
   ```
   docker container ls
   ```
7. Check that crontable into the container is set up as you expected:
   ```
   docker exec -ti [your container id] bash -c "cat /var/log/cron.log"
   ```
8. Check log when it'll be time to tun the script
   ```
   docker exec -ti [your container id] bash -c "crontab -l"
   ```

## How to run this script locally:

1. Install/check general requirements:
   ```
   ffmpeg python3
   ```

2. Install/check python3 requirements with `pip3`:
   ```
   pip3 install numpy scipy matplotlib python-telegram-bot
   ```

3. Now you can run this script!
   
   With the following command you can run the simulation with default params: 
   ```
   ./generate_3_body_simulation.py
   ```
   Without providing arguments `-T` and `-N` it will run without Telegram bot functionality.

   To modify the simulation, you can use the following arguments:
   
   `--dt DT` for simulation step;
   
   `--fps FPS` for frames per second in the video;

   `--n-bodies N_BODIES`  number of bodies (tested with up to 60);

   `--duration DURATION`  simulation duration;

   `--video-duration` video duration (if simulation duration is bigger, the visualization will be accelerated);

   `--min-score MIN_SCORE` Minimal "interest-ness" score; 
   
   `--max-score MAX_SCORE` Maximal "interest" score;

   With bigger difference between `MIN_SCORE` and `MAX_SCORE`, the script will run fewer simulations before achieving appropriate one.

   Simulations with high "interest-ness" score are looking too chaotic, and simulations with "interest-ness" score are boring (almost nothing happens).

   `--max-field-size MAX_FIELD_SIZE` Maximal field size.


   





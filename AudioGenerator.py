from multiprocessing import Process
from pydub.generators import Sine
from pydub import AudioSegment
from pydub.playback import play
import pyaudio
import time
import os
import random
from copy import copy
import sys
import math
import json
from enum import Enum
import serial

"""
width of image 640, abt 600 is non-target at 18m
SOOOO our panning needs to be vvv quiet from about x [[0..250], [390..639]]
                        and vv parabolic from x [[251.. 319], [320..389]




TODO:
make the lidar readouts adjust our boundaries for WIDE_LEFT/WIDE_RIGHT, and
    sensitivity for balancing and distancing.
    EXPLANATION: At close ranges, the target will take up a larger section
        of each frame, so larger arm movement will correspond to smaller
        target displacement distances.
        Similarly, at larger ranges, the target will take up a smaller section,
            so tiny shifts in arm movement will cause huge changes in audio
            output.
    To implement this correctly, the lidar should run in its own thread
        and populate a stack of VALID (non negative 1) ranges.
        That way we can guarantee a value at any time, w/o performance hits.
    The reason for this is that at 18meter ranges, it was super hard to
        differentiate between close to bulls and slightly closer to bulls.
    The gist of this should come down to this:
        at longer ranges, we should make our classification of WIDE_LEFT and
        WIDE_RIGHT take up more space. Currently they take up together about
        1 half of the frame. At 18m, this should be closer to maybe ~70%?


"""

class AudioGenerator:

    class Classification(Enum):
        WIDE_LEFT = 'aim right'
        WIDE_RIGHT = 'aim left'
        TRACK = 'follow'
        BULLS = 'shoot'
        NONE = 'initme'

    """
    Class for generating audio feedback. Main func is AudioGenerator.run()
    The design of this class shall integrate directly with the camera code.
    This design will CONSTANTLY generate audio, even if the OpenCV performance
        is lagging. So the audio runs in sub-processes while OpenCV runs
        in the main process.

    The core logic is:
        AudioGenerator.run(current_circle)
            if current_circle != AudioGenerator.prev_circle
                kill the process which is running the previous_circles audio
                spawn a new process playing audio for current_circle
                save the process id of the new process
            else
                pass

    :center_freq -> int: If we add pitch modulation, this is our root note (A4)
    :sine_generator -> Pydub.Generator: sine tone generator
    :cycle_time -> int: how long each audio chunk is
    :prev_circle -> [int, int]: last identified target
    :prev_pid -> int: process id of most recently spawned audio process
    :boundaries -> [int, int]: x-y boundaries of the camera window
    :sleep_time -> int: time delay between starting new sound and
                        killing old process
    """
    ONLY_VERT_PACING = True
    center_freq = 440
    bulls_freq = 880

    cycle_time_max = 1000
    cycle_time_min = 200
    cycle_time_blip = 40

    thresh_balance = 0.5
    thresh_center = 0.06
    thresh_min_volume = -16.0

    sleep_time = 0.2
    
    boundaries = []
    max_displacement = 0

    prev_pid = None
    prev_type = Classification.NONE

    error_cycles = 0
    error_limit = 10

    lidar = None

    def __init__(self, boundaries):
        """
        Initialize our class, w/ the resolution of the camera image, aka
            the boundaries
        :boundaries -> [int, int]: x, y dimensions
        """
        self.boundaries = boundaries
        self.max_displacement = math.sqrt(math.pow(boundaries[0] / 2, 2) + math.pow(boundaries[1] / 2, 2))
        try:
            self.lidar = serial.Serial('/dev/ttyACM0',115200, timeout=1)
            self.lidar.write(bytes(b'\x00\x11\x01\x45'))
        except:
            pass
    
    def extend(self, soundItems):
        """
        We generate audio (two) cycles at a time, so for the audio to continue
            playing for a longer duration, we double this sound stream 10
            times.
        :soundItems -> PyAudio.AudioSegment: one sound cycle
        :return -> PyAudio.AudioSegment: 10 sound cycles
        """
        for i in range(10):
            soundItems = soundItems + soundItems
        return soundItems

    def no(self, lol=None):
        # ignore me
        pass
    
    def play(self, audio, sleep_length, classification):
        play(audio)
    
    def spawn_play(self, sound, sleep_length, classification):
        """
        Based on a sound object, spawn a new process and play that audio
        If we have another process running somewhere else, kill it so 
            we don't have two audios playing at once
        Save the pid of the new process so we can kill it later if need be
        :sound -> PyAudio.AudioSegment: spatialized & extended sound cycles
        """
        p = Process(target=self.play, args=(sound, sleep_length, classification ))
        p.start()
        self.kill_process(self.prev_pid)
        self.prev_pid = p.pid
    
    def kill_process(self, pid):
        if pid is not None:
            time.sleep(self.sleep_time)
            try:
                os.kill(pid, 1)
            except:
                pass
        else:
            pass

    def get_balance(self, balance):
        balance_ = 0
        if balance > self.thresh_balance:
            # between 1 <-> .5             - hard right
            balance_ = 1.0
        elif balance < - self.thresh_balance:
            # between -1 <-> -0.5           - hard left
            balance_ = -1.0
        else:
            # between -0.5 <-> 0.5          - actually do some pan logic
            if balance >= - self.thresh_center and balance <= self.thresh_center:
                # between -0.02 <-> 0.02    - close enough to bulls
                balance_ = 0
            elif balance < - self.thresh_center:
                # between -0.5 <-> -0.02    -   fancy pan left
                balance_ = ( balance + self.thresh_center ) * ( self.thresh_balance / ( self.thresh_balance - self.thresh_center )) * ( 1 / self.thresh_balance)
                balance_ = - math.pow(- balance_, (1 / 2)) #TODO: maybe 1/3? or more complex
            elif balance > self.thresh_center:
                # between 0.5 <-> 0.2       - fancy pan right
                balance_ = ( balance - self.thresh_center ) * ( self.thresh_balance / ( self.thresh_balance - self.thresh_center )) * ( 1 / self.thresh_balance)
                balance_ = math.pow(balance_, (1 / 2)) #TODO: maybe 1/3? or more complex
        return balance_

    def get_distance(self, distance, balance):
        if self.ONLY_VERT_PACING:
            return math.pow(distance / ( self.boundaries[1] ) / .4, 0.8)
        else:
            center_region = 0.375 # hardcoded, sry. it basically makes it so that dist from center is 1 at the WIDE_{LEFT/RIGHT} and 0 at center
            return math.pow(( distance / self.max_displacement) / center_region, 0.8)
    
    def classify(self, balance, volume, dist_from_center):
        if balance <= -1.0:
            return self.Classification.WIDE_LEFT
        elif balance >= 1.0:
            return self.Classification.WIDE_RIGHT
        else:
            if not self.ONLY_VERT_PACING:
                if dist_from_center <= 10:
                    return self.Classification.BULLS
                else:
                    return self.Classification.TRACK
            else:
                if abs(balance) < self.thresh_center and dist_from_center < 10:
                    return self.Classification.BULLS
                else:
                    return self.Classification.TRACK
    
    def generate_beeps(self, tone, balance, volume, dist_from_center, classification):
        if classification is self.Classification.WIDE_LEFT:
            return tone.to_audio_segment(self.cycle_time_min, volume=volume)
        elif classification is self.Classification.WIDE_RIGHT:
            return tone.to_audio_segment(self.cycle_time_min, volume=volume)
        elif classification is self.Classification.TRACK:
            dist_from_center = math.pow(dist_from_center, 2)
            return tone.to_audio_segment(self.cycle_time_blip, volume=volume) + tone.to_audio_segment((self.cycle_time_max - self.cycle_time_blip) * dist_from_center, volume=-9999)

    def generate_sound(self, balance, volume, dist_from_center, classification):
        if classification is self.Classification.WIDE_LEFT:
            return self.generate_beeps(Sine(self.center_freq), balance, volume, dist_from_center, classification).pan(-1)
        elif classification is self.Classification.WIDE_RIGHT:
            return self.generate_beeps(Sine(self.center_freq), balance, volume, dist_from_center, classification).pan(1)
        elif classification is self.Classification.TRACK:
            return self.generate_beeps(Sine(self.center_freq), balance, volume, dist_from_center, classification).pan(balance)
        elif classification is self.Classification.BULLS:
            return Sine(self.bulls_freq).to_audio_segment(self.cycle_time_min, volume=volume/2)
    
    def process_circle(self, circle):
        _balance = ( circle[0] - ( self.boundaries[0] / 2 )) / ( self.boundaries[0] / 2) # -1.0 = Left, +1.0 = Right, 0.0 = Center
        _volume = abs(_balance) # 1.0 = Left, 1.0 = Right, 0.0 = Center
        if self.ONLY_VERT_PACING:
            _dist_from_center = abs( circle[1] - ( self.boundaries[1] / 2 ) )
        else:
            _dist_from_center = math.sqrt(math.pow(circle[0] - (self.boundaries[0] / 2 ), 2) + math.pow(circle[1] - ( self.boundaries[1] / 2 ), 2))
        balance = self.get_balance(_balance)
        volume = self.thresh_min_volume - ( ( 1 - math.sqrt( self.get_balance( _volume ) ) ) * self.thresh_min_volume)
        distance = self.get_distance(_dist_from_center, balance)
        classification = self.classify(balance, volume, _dist_from_center)
        return balance, volume, distance, classification

    def get_range(self):
        try:
            return int(str(self.lidar.read(20)).split('\\n')[2])
        except:
            return -1

    def run(self, circle):
        """
        balance:
            balance ranges from -1.0 <-> 1.0
            if the target is in the outer 2 quarters of the image, pan it aggressively to that side [-1, 1]
            if it is in the center 2 quarters, linearly increase/decrease from [-1<->0, 0<->1]
        volume:
            volume ranges from 0 <-> 1
            if the target is in the outer 2 quarters of the image, set volume to our min
            if it is in the center 2 quarters, exponentially increase/decrease from [-1<->0, 0<->1]
        dist_from_center:
            simple linear displacement from center accounting for x and y axis
        """
        # if not self.args["lidar"]:
        target_range = self.get_range()
        print("Current distance: " + str(target_range / 1000) + " meters")
        if circle[0] is 0 and circle[1] is 0:
            if self.error_cycles >= self.error_limit:
                if self.error_cycles == 10:
                    print("Error: no fresh circles w/in last " + str(self.error_limit) + " cycles. Killing all audio")
                self.kill_process(self.prev_pid)
                self.prev_pid = None
                self.prev_type = self.Classification.NONE
                self.error_cycles += 1
            else:
                self.error_cycles += 1
                print("Error: no circle detected, error count is now " + str(self.error_cycles))
        else:
            balance, volume, dist_from_center, classification = self.process_circle(circle)
            if classification is not self.Classification.TRACK and classification is self.prev_type:
                print("Circle found [" + str(circle[0]) + ", " + str(circle[1]) + "], but  matches previous type: " + str(classification) + ", so continuing last audio")
            else:
                print("Circle found [" + str(circle[0]) + ", " + str(circle[1]) + "] - type: " + str(classification))
                self.spawn_play(self.extend(self.generate_sound(balance, volume, dist_from_center, classification)), dist_from_center, classification)
                self.prev_type = classification
                self.error_cycles = 0

if __name__ == "__main__":
    """
    Just testing. Bounces the circle around linearly, w/ random sleeps
        to simulate OpenCV lagging
    """
    au = AudioGenerator([640, 480])
    curCircle = [150, 70]
    curIter = 10
    curIterY = 10
    while True:
        au.run(curCircle)
        time.sleep(random.uniform(0.9, 3))
        if (curCircle[0] == 640 and curIter > 0) or (curCircle[0] == 0 and curIter < 0):
            curIter = -curIter
        if (curCircle[1] == 480 and curIterY > 0) or (curCircle[1] == 0 and curIterY < 0):
            curIterY = -curIterY
        curCircle[0] += curIter
        curCircle[1] += curIterY

from Perf_Framework import ModelInference 
#from EL_Test import get_completion_from_messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set to '2' to suppress informational messages
import warnings
warnings.filterwarnings('ignore')
import time
import openai
from elevenlabs import generate, play, set_api_key, stream
import threading



openai.api_key  = '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'
set_api_key("6193d79da37c15477df5f9837d04d9a5")


def get_completion_from_messages(messages, model="gpt-4", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
       
    )
#     print(str(response.choices[0].message))
    return response.choices[0].message["content"]



def write (prompt: str):
    for chunk in openai.ChatCompletion.create(
    model= 'gpt-4',
    messages= [{'role':'system','content':prompt}],
    stream= True,
    ):
        if(text_chunk := chunk['choices'][0]['delta'].get('content')) is not None:
            yield text_chunk



def audio_output(response):
    audio_stream = generate(text=response, stream=True)
    stream(audio_stream)


def print_output():

    for output_chunk in write(f"""        
    You are the moderator of a framework concern with the prediction and detection of outliers and failures in PV systems, 
    the framework is composed of three phases, prediction to predict the generation of the PV plant taking irradiance data as input, 
    then detection by comparing the actual generation of the PV plant with the predicted generation, 
    and the points which exceeds the threshold would be labelled as outliers.I'll pass to you the results of each phase so you would comment and think out loud.
    Now let's start, the irradiance data which was the input to prediction is as following {data}, the prediction of the
    generation is as following {predictions}, the actual generation values of the PV plant are as following {actual_generation}, and finally the detected outliers {outliers}.
    You have to comment on the results of the framework and based on your exeperience check if they makes sense or not. Mentioning which ones you believe they are outliers.

    """):
        

        audio_output(output_chunk)

        for char in output_chunk:
            print(char, end="", flush=True)  # flush=True ensures the character is printed immediately
            time.sleep(0.05)  # delay for 0.05 seconds; adjust as necessary for desired "typing" speed
            
            # Check for newline and skip the next character if it's also a newline
            if char == "\n" and output_chunk:
                next_index = output_chunk.index(char) + 1
                if next_index < len(output_chunk) and output_chunk[next_index] == "\n":
                    continue




# Assuming you have the generate and stream functions already imported or defined

def print_char_by_char(input_string, delay=0.05):
    time.sleep(11.5)
    for char in input_string:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()






def output_graph():

    original_data = [2880.9312, 3427.434, 3510.3835, 3387.1768, 4276.2123, 4634.7714]
    new_data = [2069.152, 3258.4622, 3513.2542, 3237.803, 3890.261, 4388.138]

    all_data = original_data + new_data
    min_value = min(all_data)
    max_value = max(all_data)

    # Normalize data to fit between 0 and 1
    def normalize(data, min_val, max_val):
        return [(value - min_val) / (max_val - min_val) for value in data]

    normalized_original = normalize(original_data, min_value, max_value)
    normalized_new = normalize(new_data, min_value, max_value)

    # Generate ASCII plot with axes
    height = 20
    width = len(original_data)
    plot = [[' ' for _ in range(width + 3)] for _ in range(height + 1)]  # +3 for y-axis and padding, +1 for x-axis

    # Add the data points for original_data
    for idx, value in enumerate(normalized_original):
        plot_row = int((1 - value) * height)
        plot[plot_row][idx + 2] = '*'

    # Add the data points for new_data
    for idx, value in enumerate(normalized_new):
        plot_row = int((1 - value) * height)
        plot[plot_row][idx + 2] = 'o'

    # Add axes
    for row in plot:
        row[1] = '|'
    plot[-1] = ['-' for _ in range(width + 25)]
    plot[-1][1] = '+'
    plot[-1][0] = '|'

    # Display the plot
    for row in plot:
        print(''.join(row))



model_path = 'Trained_model'
scaler_path = 'Scaler\scaler.pkl'
weather_data_path = 'Irradiance_sample.parquet'
actual_generation_path = 'actual_gen.parquet'

inference_pipeline = ModelInference(model_path, scaler_path, weather_data_path, actual_generation_path)
data = inference_pipeline.load_weather_data(weather_data_path)
inference_pipeline.prepare_data(input_steps=10)
predictions = inference_pipeline.predict()
actual_generation = inference_pipeline.load_actual_gen(actual_generation_path)
residuals, outliers = inference_pipeline.detect_outliers()

#Starting generating words

def play_audio(audio):
    play(audio)

def print_char_by_char(input_string, delay=0.05):
    for char in input_string:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()



completion_one = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"system", "content":f"""

You are the a Data Scientist called Mark, you are part of a framework which was designed by EURAC Research Center.
the framework is concern with the prediction and detection of outliers and failures in PV systems, 
the framework is composed of three phases, prediction to predict the generation of the PV plant taking irradiance data as input, 
then detection by comparing the actual generation of the PV plant with the predicted generation, 
and the points which exceeds the threshold would be labelled as outliers.
You are responsible of the first two phases, I'll pass to you the results of these two phases
so you would comment and think out loud. Now let's start, the irradiance data which was the input to prediction is as following {data}, 
the prediction of the generation is as following {predictions}.
Start by introducing yourself then provide the proper information.

"""
}])





First_Response = completion_one.choices[0].message.content
First_audio = generate(First_Response, voice= 'Mark R')

fixing_threshold = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"system", "content":f"""

You are an mathematician called Maya, which works in a team to detect for outliers in PV systems.                                                                          
You will be given the actual values of the power generation, and the absolute errors between predicted and actual PV generation values.
The actual generated power values are {actual_generation} 
{residuals} Use the statistical methods:
the Median Absolute Deviation (MAD) to Calculate the threshold and output this threshold, use a constant of 3 to compute the threshold
after computing MAD.
Try to be brief when doing the math, but you must report the final result of the threshold at the end.
Start by introducing yourself first, then start computing the threshold. Report all the numbers in your response with two
decimal points.  Report all the numbers in your response with two
decimal points.
"""
}])

threshold_Response = fixing_threshold.choices[0].message.content
Threshold_audio = generate(threshold_Response, voice= 'Maya')

    
extract_threshold = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"system", "content":f"""

You will receive a text from a mathematician for the calculation of threshold to detect for outliers in PV system, 
I want you to extract the threshold and output it. 
One example is like that:
input: The median absolute error = (149.37 + 246.63) / 2 = 198
Absolute deviations from the median are (|811.78-198|, |168.97-198|, |2.87-198|, |149.37-198|, |385.95-198|, |246.63-198|) which results in
(613.78, 29.03, 195.13, 48.63, 187.95, 48.63).
The median of these absolute deviations is the MAD, which results in 98.63.
A common way to define outliers with MAD is to set the threshold as some factor, 
typically 3, times the MAD. So, our upper limit threshold according to MAD is 494.89.
Between the Mean-based approach and Median Absolute Deviation (MAD),
MAD tends to be a more robust method when dealing with outliers as it isn’t affected as much by extremely large or small values. 
Therefore, I would suggest we set the threshold upper limit at 494.89 (MAD approach). 
your output should be: 494.89      
Now the text you have to extract the threshold from is{threshold_Response}.                                                                                                                                                                                                                                                                                                   
"""
}])

threshold_value = extract_threshold.choices[0].message.content



completion_two = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"system", "content":f"""

You are Laurance, the moderator of a framework concern with the prediction and detection of outliers and failures in PV systems, 
the framework is composed of three phases, prediction to predict the generation of the PV plant taking irradiance data as input, 
then detection by comparing the actual generation of the PV plant with the predicted generation, 
and the points which exceeds the threshold would be labelled as outliers.
Your colleagues Mark takes care of the prediction phase, where the other colleauge Maya calculates the threshold value to 
detect for outliers.
I'll pass to you the results of each phase so you would comment and think out loud, I will also give you threshold which
was computed by our mathematecian Maya, and you have to detect the outlier based on this threshold
which you can apply on the residual between the predicted and the actual values.
the prediction of the generation is as following {predictions}, 
the actual generation values of the PV plant are as following {actual_generation}, the residuals are {residuals} and finally the threshold {threshold_value}.
You have to comment on the results of the framework 
and based on your exeperience check if they makes sense or not. 
Mentioning which ones you believe they are outliers.
Start by introducing yourself first as Laurance. Report all the numbers in your response with two
decimal points.
"""
}])


Last_response = completion_two.choices[0].message.content
Last_audio = generate(Last_response, voice= 'Laurance')



def play_audio(audio):
    play(audio)

def print_char_by_char(input_string, delay=0.06):
    for char in input_string:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()




introduction = "Welcome to HelioHealth: A Diagnostic Framework for the Solar System. This system \
is meticulously overseen by three robotic experts. Mark, our data scientist, \
delves into and predicts from the data. Maya, our mathematician, ensures precise calculations for the detection design.\
Lastly, Laurance serves as the primary moderator for the framework"

next_step = 'HelioHealth is activated now, new data has been logged from the PV plant.'


Outro = "That's the latest update on the plant as provided by HelioHealth. \
If you encounter any technical challenges with the framework, please contact our institute through their website: www.eurac.edu"

Introduction_audio = generate(introduction, voice= 'Laurance')

next_step_audio = generate(next_step, voice= 'Laurance')

outro_audio = generate(Outro, voice='Laurance')

time.sleep(3)
# Create and start threads for the introduction
introduction_thread_audio = threading.Thread(target=play_audio, args=(Introduction_audio,))
introduction_thread_print = threading.Thread(target=print_char_by_char, args=(introduction,))
introduction_thread_audio.start()
introduction_thread_print.start()

# Wait for the introduction threads to complete
introduction_thread_audio.join()
introduction_thread_print.join()
print('---------------------------------------------------------------------------')


output_graph()


print('---------------------------------------------------------------------------')
time.sleep(3)

# Create threads for the two functions
next_step_thread_audio = threading.Thread(target=play_audio, args=(next_step_audio,))
next_step_thread_print = threading.Thread(target=print_char_by_char, args=(next_step,))
next_step_thread_audio.start()
next_step_thread_print.start()

next_step_thread_audio.join()
next_step_thread_print.join()

print('---------------------------------------------------------------------------')

time.sleep(3)

First_Response_thread_audio = threading.Thread(target=play_audio, args=(First_audio,))
First_Response_thread_print = threading.Thread(target=print_char_by_char, args=(First_Response,))
First_Response_thread_audio.start()
First_Response_thread_print.start()

First_Response_thread_audio.join()
First_Response_thread_print.join()

print('---------------------------------------------------------------------------')

time.sleep(3)


Threshold_audio_thread_audio = threading.Thread(target=play_audio, args=(Threshold_audio,))
Threshold_Response_thread_print = threading.Thread(target=print_char_by_char, args=(threshold_Response,))
Threshold_audio_thread_audio.start()
Threshold_Response_thread_print.start()

Threshold_audio_thread_audio.join()
Threshold_Response_thread_print.join()

print('---------------------------------------------------------------------------')

time.sleep(3)


Last_audio_thread_audio = threading.Thread(target=play_audio, args=(Last_audio,))
Last_Response_thread_print = threading.Thread(target=print_char_by_char, args=(Last_response,))
Last_audio_thread_audio.start()
Last_Response_thread_print .start()

Last_audio_thread_audio.join()
Last_Response_thread_print.join()
time.sleep(1)
print('---------------------------------------------------------------------------')

Outro_audio_thread_audio = threading.Thread(target=play_audio, args=(outro_audio,))
Outro_Response_thread_print = threading.Thread(target=print_char_by_char, args=(Outro,))
Outro_audio_thread_audio.start()
Outro_Response_thread_print .start()



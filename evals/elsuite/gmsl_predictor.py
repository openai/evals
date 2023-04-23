#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Predict Global Mean Sea Level (GMSL) variation with Global Isometric Adjustment (GIA).
# GMSL refers to the average height of the sea surface across the entire globe.
# GIA is the response of the Earth's crust to the redistribution of mass due to changes in ice mass on land.
# This change in ice mass impacts global sea level; GMSA with GIA refers to Global Mean Sea Level variation
# with the results of those changes subtracted to reflect the actual sea level change by other factors.
# This eval attempts to determine the accuracy of predicting GMSL with and without GIA.
# The version below provides brief context about what the numbers mean. GPT-3.5 and 4 are both fully knowledgeable
# of the terms used. The number returned by the model should be the GMSL with the GIA subtracted, in essence
# predicting the GIA blind.


# In[ ]:


# The GMSL data was generated using the Integrated Multi-Mission Ocean Altimeter Data for
# Climate Research (http://podaac.jpl.nasa.gov/dataset/MERGED_TP_J1_OSTM_OST_ALL_V51). It combines
# Sea Surface Heights from the TOPEX/Poseidon, Jason-1, OSTM/Jason-2, Jason-3, and Sentinel-6 Michael Freilich
# missions to a common terrestrial reference frame with all inter-mission biases, range and geophysical corrections
# applied and placed onto a georeferenced orbit.  This creates a consistent data record throughout
# time, regardless of the instrument used.  Note, the most recent estimates of GMSL (post March 28, 2022)
# derived from the Sentinel-6 Michael Freilich mission are preliminary as validation and
# reprocessing procedures for Sentinel-6 are ongoing.


# In[ ]:


# Data source and source of above information block:
# GSFC. 2021. Global Mean Sea Level Trend from Integrated Multi-Mission Ocean Altimeters TOPEX/Poseidon
# Jason-1, OSTM/Jason-2, and Jason-3 Version 5.1. Ver. 5.1 PO.DAAC, CA, USA.
# Dataset accessed [2023-04-20] at https://doi.org/10.5067/GMSLM-TJ151.


# In[ ]:


# Download data set at NASA Earthdata Search. Collection name:
# Global Mean Sea Level Trend from Integrated Multi-Mission Ocean Altimeters TOPEX/Poseidon, Jason-1, OSTM/Jason-2, and Jason-3 Version 5.1


# In[1]:


import os
import random
import re
from math import sqrt

import openai
from sklearn.metrics import r2_score

import evals
import evals.metrics

# In[4]:


# This is the main class that is referenced in gmsl_predictor.yaml and is used by oaievals.py.
class GMSLPredictor(evals.Eval):
    def __init__(
        self, train_jsonl, test_jsonl, train_samples_per_prompt=10, completion_fns=None, **kwargs
    ):
        super().__init__(completion_fns=completion_fns, **kwargs)
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl
        self.train_samples_per_prompt = train_samples_per_prompt
        self.api_key = os.environ["OPENAI_API_KEY"]
        openai.api_key = self.api_key

    # This function is called by the run function of oaieval.py.
    def run(self, recorder):
        # Define the objects I used to later get R^2 data.
        actual_gmsl_list = []
        predicted_gmsl_list = []
        # Define the samples and send the test samples to eval_all_samples in eval.py.
        self.train_samples = evals.get_jsonl(self.train_jsonl)
        test_samples = evals.get_jsonl(self.test_jsonl)
        results = self.eval_all_samples(recorder, test_samples)

        # Append the results to my lists of ideal and predicted values for R^2 calculation.
        for result in results:
            actual_gmsl_list.append(result["act_gmsl"])
            predicted_gmsl_list.append(result["pred_gmsl"])

        # Sum up the MSE and MAE values from the samples and get averages. Perform R^2 calculation.
        mse_sum = sum(result["mse"] for result in results)
        mae_sum = sum(result["mae"] for result in results)
        rmse_sum = sum(result["rmse"] for result in results)
        r2 = r2_score(actual_gmsl_list, predicted_gmsl_list)

        n_samples = len(results)

        mse_avg = mse_sum / n_samples
        mae_avg = mae_sum / n_samples
        rmse_avg = rmse_sum / n_samples

        # Returns list of dictionary objects to run function of oaieval.py.
        return {"mse": mse_avg, "mae": mae_avg, "rmse": rmse_avg, "r2": r2}

    # This is a function to create an instance of the CompletionFn class in api.py. Called from eval_sample.
    def query_model(self, prompt):
        response = self.completion_fn(prompt=prompt)
        # Get and return the resulting sample from the AI.
        return response.get_completions()[0]

    # This function evaluates a single sampled. It is called from eval_all_samples in eval.py. It returns a dictionary
    # object of recorded metrics which are passed back to eval_all_samples which maintains a list of these dictionary
    # objects. The return of this function will ultimately be passed to this file's run function.
    def eval_sample(self, test_sample, rng: random.Random):
        # Stuffing and prompt creation.
        stuffing = rng.sample(self.train_samples, self.train_samples_per_prompt)
        prompt = [
            {
                "role": "system",
                "content": "You will be given Global Mean Sea Level (GMSL) variation WITHOUT"
                " Global Isostatic Adjustment in mm with respect to TOPEX/Jason collinear mean reference."
                " You will also be given the year and fraction of year in decimal form of when the measurement happened."
                " Predict the global mean sea level variation WITH global isostatic adjustment (GMSL with GIA)"
                ", in mm, with respect to TOPEX/Jason collinear mean reference."
                " Your response should ONLY include the predicted GMSL with GIA in mm.",
            },
            {
                "role": "user",
                "content": "Date of Measurements: 1993.011526 GMSL without GIA: -38.61",
            },
            {"role": "assistant", "content": "-38.64"},
            {
                "role": "user",
                "content": "Date of Measurements: 2010.794397 GMSL without GIA: 14.24",
            },
            {"role": "assistant", "content": "18.6"},
            {
                "role": "user",
                "content": "Date of Measurements: 2002.025209 GMSL without GIA: -15.75",
            },
            {"role": "assistant", "content": "-13.05"},
        ]
        for i, sample in enumerate(stuffing + [test_sample]):
            if i < len(stuffing):
                prompt += [
                    {"role": "system", "content": sample["input"]},
                    {"role": "system", "content": sample["ideal"]},
                ]
            else:
                prompt += [{"role": "user", "content": sample["input"]}]
        # Call my query_model function and get the result.
        model_prediction = self.query_model(prompt)
        # Assign this sample's ideal value.
        actual_gmsl = float(test_sample["ideal"])
        # GPT 3.5 really likes to add all sorts of numbers to the response. We need to parse the correct one.
        # This regex gets all the numbers in the sample response.
        matches = re.findall(r"[-]?\d*\.\d+|[-]?\d+", model_prediction)
        # If there's at least one number, do the following:
        if len(matches) >= 1:
            # Convert matches to float
            matches = [float(match) for match in matches]

            # Remove the actual_gmsl from the list if there are multiple numbers under 1000
            if sum(1 for match in matches if match < 1000) > 1:
                matches = [match for match in matches if match != actual_gmsl]

            if len(matches) >= 1:
                # Assign the smallest number in the list to predicted_gmsl
                predicted_gmsl = min(matches)
            else:
                predicted_gmsl = 0
        else:
            predicted_gmsl = 0
        # Calculate (root) mean squared error ([r]mse) and mean absolute error (mae) below.
        mse = (predicted_gmsl - actual_gmsl) ** 2
        mae = abs(predicted_gmsl - actual_gmsl)
        rmse = sqrt(mse)

        # Define metrics and return to eval_all_samples, to ultimately be used in my run function.
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "pred_gmsl": predicted_gmsl,
            "act_gmsl": actual_gmsl,
        }

        return metrics


# In[ ]:

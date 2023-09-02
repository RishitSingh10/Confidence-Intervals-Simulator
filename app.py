import matplotlib
matplotlib.use('Agg')  # Use the "Agg" backend

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFProtect
from wtforms import FloatField, IntegerField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
import os
import secrets

secret_key = secrets.token_hex(16)  # Generate a 32-character (16 bytes) hex secret key
print("Generated Secret Key:", secret_key)

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = secret_key
csrf = CSRFProtect(app)

class SimulationForm(FlaskForm):
    population_mean = FloatField('Population Mean (μ):', validators=[DataRequired()])
    population_sd = FloatField('Population SD (σ):', validators=[DataRequired()])
    sample_size = IntegerField('Sample Size (n):', validators=[DataRequired()])
    confidence_level = FloatField('Confidence Level:', validators=[DataRequired()])
    num_intervals = IntegerField('Number of Intervals:', validators=[DataRequired()])
    submit = SubmitField('Simulate')

def calculate_confidence_intervals(population_mean, population_sd, sample_size, confidence_level, num_intervals):
    z_value = norm.ppf(1 - (1 - confidence_level) / 2)
    sem = population_sd / np.sqrt(sample_size)

    samples = np.random.normal(population_mean, population_sd, (num_intervals, sample_size))
    sample_means = samples.mean(axis=1)
    margin_of_error = z_value * sem

    confidence_intervals = [(sample_mean - margin_of_error, sample_mean + margin_of_error) for sample_mean in sample_means]

    print(f"Population Mean: {population_mean}")
    print(f"Sample Means: {sample_means}")
    print(f"Confidence Intervals: {confidence_intervals}")

    return sample_means, confidence_intervals

@app.route('/', methods=['GET', 'POST'])
def index():
    form = SimulationForm()
    image_path = None  # Initialize image_path
    percentage_containing_mean = None  # Initialize percentage_containing_mean
    population_mean = None

    if form.validate_on_submit():
        population_mean = form.population_mean.data
        population_sd = form.population_sd.data
        sample_size = form.sample_size.data
        confidence_level = form.confidence_level.data
        num_intervals = form.num_intervals.data

        sample_means, confidence_intervals = calculate_confidence_intervals(population_mean, population_sd, sample_size, confidence_level, num_intervals)
        num = 0
        plt.figure(figsize=(6, 8))
        for i in range(len(sample_means)):
            if (confidence_intervals[i][0] <= population_mean <= confidence_intervals[i][1]):
                num += 1
                plt.errorbar(x=sample_means[i], y=i, xerr=2 * population_sd / (np.sqrt(sample_size)), yerr=0.0,
                             linestyle='', c='green', lolims=True)
                print(f"Interval {i}: Green")
            else:
                plt.errorbar(x=sample_means[i], y=i, xerr=2 * population_sd / (np.sqrt(sample_size)), yerr=0.0,
                             linestyle='', c='red', lolims=True)
                print(f"Interval {i}: Red")

            plt.axvline(x=population_mean, ymin=0, ymax=num_intervals, color='blue', linestyle='--')

        percentage_containing_mean = num / len(sample_means)
        
        plt.xlabel(f'Mean of Sample Means: {sample_means.mean()}')
        plt.xlim(population_mean - 10, population_mean + 10)
        plt.ylabel('Num of intervals')
        plt.title('Sample Means and Confidence Intervals')

        image_path = os.path.join(app.root_path, 'static', 'confidence_intervals.png')
        plt.savefig(image_path)
        plt.close()

    return render_template('index.html', form=form, image_path=image_path, percentage_containing_mean=percentage_containing_mean, population_mean=population_mean)

if __name__ == '__main__':
    app.run(port=4996, debug=True)

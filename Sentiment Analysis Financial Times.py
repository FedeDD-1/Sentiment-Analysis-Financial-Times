"""
This code performs sentiment analysis on news articles from the
Financial Times using NLTK's sentiment analyzer. The objective is to plot the 
moving average of a sentiment score over time to investigate whether the average 
sentiment changed during the course of the financial crisis (2007-2009).
It processes the dataset by merging titles and texts, computes sentiment scores, 
and applies a moving average to smooth trends over time.
The results are visualized with a time-series plot, highlighting the sentiment 
trends from 2007 to 2009. The approach follows an object-oriented structure, 
performing data loading, processing, analysis, and visualization within a 
reusable class. 

@author: Federico Durante
"""

"""Import libraries"""
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

"""Class for sentiment analysis"""
class Sentiment_Analysis:
    """Class for performing sentiment analysis on the dataset"""
    
    def __init__(self, file_path, window=100): 
        """
        Constructor method. Handles data loading and preprocessing.

        Parameters
        ----------
        file_path : str
            Path to the dataset file
        window : int, optional
            Window size for moving average computation (default is 100)
        
        Returns
        -------
        None
        """
        self.file_path = file_path
        self.window = window
        self.df = None
        self.analyzer = SentimentIntensityAnalyzer()
        self.load_and_process_data()

    def load_and_process_data(self):
        """
        Method to load and preprocess the dataset
        
        Returns
        -------
        None
        """
        self.df = pd.read_pickle(self.file_path, compression="gzip")
        
        """Drop rows with missing values"""
        self.df.dropna(subset=["Title", "Text", "Date"], inplace=True)
        
        """Combine Title and Text columns into a single column"""
        self.df["Title and text"] = self.df["Title"] + " " + self.df["Text"]
        
        """Convert Date column to datetime format"""
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        
        """Filter dataset to include only data from 2007 to 2009"""
        self.df = self.df[(self.df["Date"] >= "2007-01-01") & (self.df["Date"] <= "2009-12-31")]
        
        """Sort the dataset by Date"""
        self.df.sort_values(by="Date", inplace=True)

    def compute_sentiment(self):
        """
        Method for computing sentiment scores using NLTK's analyzer
        
        Returns
        -------
        None
        """
        self.df["Sentiment Score"] = self.df["Title and text"].apply(lambda text: self.analyzer.polarity_scores(text)['compound'])

    def compute_moving_average(self):
        """
        Method for computing the moving average of sentiment scores
        
        Returns
        -------
        None
        """
        self.df["Sentiment MA"] = self.df["Sentiment Score"].rolling(window=self.window, min_periods=self.window).mean()

    def plot_sentiment(self):
        """
        Method for plotting the sentiment trend over time
        
        Returns
        -------
        None
        """
        plt.figure(figsize=(12, 6))
        
        """Plot the moving average sentiment score over time"""
        plt.plot(self.df["Date"], self.df["Sentiment MA"], label="Moving Average Sentiment", color="blue")
        
        """Label the axes"""
        plt.xlabel("Date (Year-Month)")
        plt.ylabel("Sentiment Score (Moving Average)")
        
        """Add the title"""
        plt.title("Sentiment Analysis of Financial Times News (2007-2009)")
        
        """Add legend"""
        plt.legend()
    
        """Add grid"""
        plt.grid(True, linestyle='--', alpha=0.5)
        
        """Save figure"""
        plt.savefig("CE2-A5_Federico_Durante.pdf")
        
        """Show the plot"""
        plt.show()

    def run_analysis(self):
        """
        Method for executing the sentiment analysis
        
        Returns
        -------
        None
        """
        self.compute_sentiment()
        self.compute_moving_average()
        self.plot_sentiment()

"""Main entry point"""
if __name__ == "__main__":
    
    """Define the file path"""
    file_path = "ft-articles.pkl.tar.gz"
    
    """Run sentiment analysis"""
    sentiment_analysis = Sentiment_Analysis(file_path, window=500)
    sentiment_analysis.run_analysis()

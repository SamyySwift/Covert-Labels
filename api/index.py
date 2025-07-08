from flask import Flask
import sys
import os

# Add the parent directory to the path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

if __name__ == "__main__":
    app.run()
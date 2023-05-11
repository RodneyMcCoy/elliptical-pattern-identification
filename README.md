<a name="readme-top"></a>




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/RodneyMcCoy/elliptical-pattern-identification">
    <img src="https://github.com/RodneyMcCoy/elliptical-pattern-identification/blob/master/res/logo.ico" alt="Logo" width="120">
  </a>

<h3 align="center">Elliptical Pattern Identification</h3>

  <p align="center">
    This project host's our work for identifying Gravitational Waves via Radiosonde data. Check out this README for further details.
  </p>
</div>



<!-- TABLE OF CONTENTS -->

### Table of Contents
<ol>
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#downloading-the-application"> Downloading the Application</li>
      <li><a href="#using-the-application"> Using the Application </li>
      <li><a href="#want-to-build-the-project-yourself"> Want To Build the Project Yourself?</li>
    </ul>
  </li>
  <li><a href="#bugs">Bugs</a></li>
  <li><a href="#notes-to-future-contributors">Notes To Future Contributors</a></li>
  <li><a href="#contributors">Contributors</a></li>
  <li><a href="#acknowledgments">Acknowledgments</a></li>
  <li><a href="#license">License</a></li>
</ol>






<!-- GETTING STARTED -->
## Getting Started

<!-- DOWNLOADING THE APPLICATION -->
### Downloading the Application

1. Navigate to [Releases](https://github.com/RodneyMcCoy/elliptical-pattern-identification/releases) to download the application on your own machine.
2. You should see a list of folders starting with `ApplicationForOS` where `OS` is the operating system for the application. Download the folder for your specific operating system.
3. Once you have downloaded the folder, extract the contents.
4. Once you open the now extracted folder, select the dist folder. You should be able to see an application named `EllipticalPatternIdentification` with some operating system specific file extension like `.exe`. You can now open the application



### Using the Application

1. Once you open the `EllipticalPatternIdentification` application, you should be able to see buttons to select files to process, along with various parameter inputs for the algorithm, and more. 
2. Once you have inputted files into the program, you can use the buttons on the sidebar to look at files and their raw data along with data from the backend algorithms.
3. To actually process files, select the button `Process Files`. Processing may take a while, the window should update occasionally with the progress of processing the current file. Stopping processing will loose the results of the current file being processed.



### Want To Build the Project Yourself?

1. Clone the repository to your local machine
2. Make sure you have the required dependencies installed. 
	- Python 3
	- All Usual Python Modules
	- Tkinter Python Module
3. We suggest using anaconda since it has all of the relevant dependencies when you use Spyder.
4. Navigate to src -> main.py
5. Run main.py with your favorite python environment
	- main.py starts up the GUI, and the GUI interfaces with all of the backend algorithms



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- utilityScripts:
	this directory contains a variety of programs I have created to test ideas, learn analysis 	techniques, and troubleshoot problems. None are essential for analysis, but provide some insight to the evolution of the main analysis script. It also contains the original hodograph code from Thomas Colligan that I ported over to Python. -->





<!-- BUGS -->
## Bugs

As this application wasn't able to be tested to its fullest extent, bugs most likely will occur. They should be listed under [issues](https://github.com/RodneyMcCoy/elliptical-pattern-identification/issues). I (Rodney) will be semi-actively monitoring this repository over the summer to account for this. You can also email me at rbmj2001@outlook.com if urgent bugs need fixing. 

<!-- NOTES-TO-FUTURE-CONTRIBUTORS -->
## Notes To Future Contributors

In the design of this repository, having as little coupling as possible between the frontend and backend was essential. The front end is important but for testing, we wanted to be able to execute the backend by itself without any other front end code. There are only two locations where the front and backend actually connect, but i think pointing them out will make it so when future changes to the backend actually happen, people will not inevitably ignore the frontend since they don't want to update it. 

1. The FileWindow class takes the data which is outputed by the back end and renders it on the screen. To be able to Cohesivly render that data, it obviously needs prior understanding of what data is outputted. To find and edit this code to render different data, navigate [here](https://github.com/RodneyMcCoy/elliptical-pattern-identification/blob/3aeb31403bed1df2107588743c6c20120d374f9d/src/FrameClasses.py#LL281C18-L281C18).

2. The BackEndInterface executes the back end. It passes a single file path to ProcessSingleFile, where the backend can be set up and run. Unfortunately, when the frontend executes the backend the backend cant open any matplotlib visuals, it will throw many errors. To find and edit this code to execute the backend in the frontend, navigate [here](https://github.com/RodneyMcCoy/elliptical-pattern-identification/blob/c64f1daa51ac8050321594a529b946198427be80/src/BackEndInterface.py#L35).

<!-- CONTRIBUTORS -->
## Contributors

Initially, this code base was created by Thomas Colligan.

Then, malachiRivkin's and kareece's expanded on Colligan's code, and ported it into Python, in their [Original Work](https://github.com/malachiRivkin/hodographAnalysis) for the 2020 solar eclipse project. To contact those two, you can *hopefully* find malachiRivkin [Here](https://github.com/malachiRivkin) and kareece [Here](https://github.com/kareece). 

We have revamped and continued their work at this repository for our *Senior Capstone Project*. Our information is below. Any questions that aren't answered here should be sent to Dr. Bernards.


**Rodney McCoy** &middot;
[Github Profile](https://github.com/RodneyMcCoy) &middot;
[Email](rbmj2001@outlook.com) &middot;
[Phone](208-860-4186)


**Riley Doyle** &middot;
[Github Profile](https://github.com/rdoyle0914) &middot;
[Email](doyl1482@vandals.uidaho.edu) &middot;
[Phone](805-850-8594)


**Luis Lopez** &middot;
[Github Profile](https://www.example.com/) &middot;
[Email](lope9245@vandals.uidaho.edu) &middot;
[Phone](208-320-2344)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
This project was also made with the assistance of Dr. Matthew Bernards and Konstantine Geranios, an Associate Professor and Graduate Student respectively, from the Chemical & Biological Engineering Department at the University of Idaho.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

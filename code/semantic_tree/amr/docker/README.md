# A Docker image for using the Spring AMR parser

A Docker image that hosts a service that parses AMR using the SPRING parser.

To build the Docker image:
1. Clone this repo: `git clone https://github.com/plandes/amrspring`
1. Set the working directory: `cd amrspring`
1. Download the model(s) from the [AMR SPRING parser] repository.
1. Change the working directory to the docker source tree: `cd docker`
1. Build the image: `cd docker ; make build`
1. Check for errors.
1. Start the image: `make up`
1. Test using a method from [usage](#usage).

Of course, the server code can be run without docker by cloning the [AMR SPRING
parser] repository and adding the [server code](src).

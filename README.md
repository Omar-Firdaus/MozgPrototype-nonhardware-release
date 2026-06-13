Background:

Mozg is a startup/side hustle that I've been running for a bit. I'm making a custom pair of smart glasses, and this software is the first ever one created for the glasses to prototype and just create an mvp.

Because this was meant to be linked to the hardware I built, I've linked a demo in the project and here: https://youtu.be/BUaTyMZ2fDE (note that this demo was shot before some changes I made, so the version you see after running it is differnet in a few ways)

Motivation:

Having a heads up hud is awesome. Really wanted to have a pair of glasses with an assistant built in, and this was the first step of that goal.

Requirments:

macOS or linux, node.js 18+, python 3.9+

cd smart-glasses
./scripts/bootstrap.sh 
npm start

How it works:

An Electron window plus some Python services on the Mac handle camera, speech, and data. For this version, almost everything runs on the machine locally, except APIs for ai, calendar, search, etc. As this release is a prototype for software that's going to run on the firmware locally, the account flow isn't too developed.

Usage:

It's not really meant for usage without the hardware, however I did make some changes to the code so you can try it out without the hardware. Here are the steps if you want to try and run it on mac.

- npm start
- hit launch hud
- tap assist
- after you exit you can view the context of specific people on the "people" page
- there's a chat on the "home" page where it has the context of all your conversations
- settings are self explanatory

Tech stack:

It's an electron app that spawns python services over websockets. Most of teh vision stuff you see is working through openCV, and the facial recegnition works through ONNX ArcFace. There's a few APIs that are connected to do some pretty cool things, they are as follows:

- Apollo to parse people's info
- Firecrawl to search the web
- Google cal for scheduling
- For AI in the app, you can choose between models from anthropic, openai, and deepseek.
- Text to speech happend through deepgram

(Do be noted that I've cancelled/run out of credits for a few of these if you are trying to get the app running on your own, so to see these features in action you can check out the demo video)

AI Usage:

I'm pretty confident that well over 70% of this project was human written. My most heavy usage of AI in this project was the UI (Specifically for the HUD's liquid glass which you can probably tell). Also used composer agent for debugging and fixing my billion api errors which may be the reason that on one of the hacaktimes you see a fat block called "browsing". 


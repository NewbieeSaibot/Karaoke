# AI Karaoke

### A project created to extend the limits of simple karaoke. 

<p>Basically, most of karaoke have only midi sounds without vocals to play while a singer is performing.
The problem is that midi audios are pretty annoying. So the main idea to solve this problem was to use unmix machine learning models to generate no vocal audios from the 
original songs. To extract the lyrics we tested speech to text machine learning models, but they performed really bad, so the solution was to extract the lyrics from websites with some sort 
of scrapping and use an algorithm to algign the extracted lyric with vocal audio. This alignment is important because its easier for singers who doesn't know the lyric to have a good
performance following the aligned lyric.</p>

### Already implemented Features

<ul>
  <li>Simple GUI with selection list for musics</li>
  <li>Music Player</li>
  <li>Auto alignment algorithm</li>
  <li>Unmix algorithm</li>
  <li>Simple Lyrics Display</li>
</ul>

### Necessary Features Not Implemented

<ul>
  <li>Music Folder Picker</li>
  <li>Lyric scrapper</li>
</ul>

### Cool Features to Implement

<ul>
  <li>Get input audio</li>
  <li>Score system based on pitching</li>
  <li>Use video to evaluate performance</li>
  <li>Use Music Transcription on vocals to show the notes for the singer</li>
</ul>


### How to Use

<ol>
    <li>Clone Repo.</li>
    <li>Install requirements.txt</li>
    <li>Build the correct folder structure with your music</li>
    <li>Change path inside main.py to the folder with your musics</li>
    <li>Run it.</li>
</ol>

#### Folder structure
<p>The folder structure is pretty simple. Basically you will need to create a folder with any name and create one folder inside it for every music you want to put on karaoke list. Example of a structure with two musics called "music_a" and "music_b":</p>
<ul>
  <li>
  MainFolder
  </li>
  <ul>
      <li>music_a
      <ul>
          <li>music_a.wav</li>
      </ul>
      </li>
      <li>music_b
      <ul>
        <li>music_b.mp3</li>
        </ul>
    </li>
  </ul>
 </ul>



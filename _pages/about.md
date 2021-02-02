---
layout: page
permalink: /
title: about
nav: about
description_lab: <a href="http://faculty.cse.tamu.edu/guni/pistar/index.html" class="page-description" target="_blank">Pi Star Lab</a>
description_dept: <a href="https://engineering.tamu.edu/cse/index.html" class="page-description" target="_blank">Computer Science and Engineering</a>
description_uni: <a href="https://www.tamu.edu/" class="page-description" target="_blank">Texas A&M University</a>
address: <a href="https://www.google.com/maps/place/Engineering+Activities+Building+C/@30.6152746,-96.3400254,17z/data=!4m12!1m6!3m5!1s0x86468390a7f39815:0xa8543fc19fb3b7a2!2sEngineering+Activities+Building+C!8m2!3d30.61527!4d-96.3378367!3m4!1s0x86468390a7f39815:0xa8543fc19fb3b7a2!8m2!3d30.61527!4d-96.3378367" class="page-description" target="_blank">EABC, Room 107B, 588 Lamar St, College Station, TX 77840</a>
---

<div class="col p-0 pt-4 pb-4">
  <h1 class="pb-3 title text-left"><span class = "font-weight-bold">Sheel</span> Dey</h1>
  <h6 class="m-0 mb-2" style="font-size: 0.83em;">{{ page.description_lab }}</h6>
  <h6 class="m-0 mb-2" style="font-size: 0.83em;">{{ page.description_dept }}</h6>
  <h6 class="m-0 mb-2" style="font-size: 0.83em;">{{ page.description_uni }}</h6>
  {% if page.address %}
      <h6 class="m-0 mb-2" style="font-size: 0.83em;">{{ page.address }}</h6>
  {% endif %}
</div>

<!-- Introduction -->

<div style="display: flex; flex-wrap: wrap;">
    <div class="text-justify p-0">
        <div class="col-xs-12 col-sm-6 p-0 pt-2 pb-sm-2 pb-4 pl-sm-4 text-center" style="float: right;">
          <img class="profile-img img-responsive" src="{{ 'prof_pic.jpg' | prepend: '/assets/img/' | prepend: site.baseurl | prepend: site.url }}">
        </div>

        <p>
            Hi, I am a 2<sup>nd</sup> year computer science Ph.D. student at <a href="http://www.cmu.edu/" target="_blank">Texas A&M University</a>, where I work with <a href="http://faculty.cse.tamu.edu/guni/" target="_blank">Dr. Guni Sharon</a> in the <a href="http://faculty.cse.tamu.edu/guni/pistar/index.html" target="_blank">Pi Star Lab</a>.

            My research interests lie at the intersection of reinforcement learning and robotics. My current research focuses on leveraging expert demonstrations and interventions to make reinforcement learning safer.
        </p>

        <p>
            I previously received an M.S. in computer science from Texas A&M University, during which I was fortunate to work with <a href="https://www.atlaswang.com/" target="_blank">Dr. Atlas Wang</a> and <a href="https://medicine.tamu.edu/faculty/wang.html" target="_blank">Dr. Jun Wang</a> on brain image analysis. I also worked with <a href="https://engineering.tamu.edu/mechanical/profiles/rathinam-sivakumar.html" target="_blank">Dr. Sivakumar Rathinam</a> developing audio-based machine learning algorithms for autonomous driving.
        </p>

    </div>
</div>

<div class="col text-justify p-0">
    
</div>

<!-- News -->
<div class="news mt-3 p-0">
  <h1 class="title mb-4 p-0">news</h1>
  {% assign news = site.news | reverse %}
  {% for item in news limit: site.news_limit %}
    <div class="row p-0">
      <div class="col-sm-2 p-0">
        <span class="badge red font-weight-bold text-uppercase align-middle date ml-3">
          {{ item.date | date: "%b %-d, %Y" }}
        </span>
      </div>
      <div class="col-sm-10 mt-2 mt-sm-0 ml-3 ml-md-0 p-0 font-weight-light text">
        <p>{{ item.content | remove: '<p>' | remove: '</p>' | emojify }}</p>
      </div>
    </div>
  {% endfor %}
</div>

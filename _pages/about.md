---
layout: about
title: About
permalink: /
description: <a href="https://s3d.cmu.edu/">Software and Societal Systems Department</a> • <a href="https://www.cs.cmu.edu/">School of Computer Science</a>  • <a href="https://www.cmu.edu/">Carnegie Mellon University</a>

profile:
  align: right
  image: syn-pic-s3d.jpg
  address: >
    <p>Office: <a href="https://goo.gl/maps/gSZmnxHUU13Deg1u7">TCS Hall</a>, Room 313</p>
    <p>Email: yiningsh at cs.cmu.edu</p>

news: false  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: true  # includes social icons at the bottom of the page

display_categories: [HCPS Lab, Mars]
horizontal: true
---

I am a first-year PhD student in Software Engineering at <a href="https://s3d.cmu.edu/">Software and Societal Systems Department</a>, <a href="https://www.cmu.edu/">Carnegie Mellon University</a>, advised by <a href="https://eskang.github.io/">Dr. Eunsuk Kang</a>. My research interests are in Software Engineering, Artificial Intelligence, Cyber-Physical Systems and Formal Methods.

My current research focuses on AI's impacts on long-term fairness, but as a new PhD student I'm exploring widely in Software Engineering.

Before joining Carnegie Mellon University, I was pursuing my bachelor of Engineering in Computer Science at <a href="https://www.shanghaitech.edu.cn/eng/">ShanghaiTech University</a>, where I was a research assistant in <a href="https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/team/">Human-Cyber-Physical Systems Lab</a>, advised by <a href="https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/">Dr. Zhihao Jiang</a>. My research at HCPS Lab focused on developing model-based intelligent assistance systems in Transportation and Healthcare. Prior to joining HCPS Lab, I was a member of <a href="https://vic.shanghaitech.edu.cn/">Visual & Data Intelligence Center</a>, advised by Prof. [Jingyi Yu](https://vic.shanghaitech.edu.cn/vrvc/en/people/jingyi-yu/), and my work primarily aimed at solving problems in *2D Face Reconstruction* by using the techniques from machine learning and computer graphics.

Please feel free to contact me through email and discuss collaboration opportunities!



<br/>
<br/>
<br/>
<br/>

## Selected Projects
<div class="projects">
  {% if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized projects -->
    {% for category in page.display_categories %}
      <h2 class="category">{{ category }}</h2>
      {% assign categorized_projects = site.projects | where: "category", category %}
      {% assign sorted_projects = categorized_projects | sort: "importance" %}
      <!-- Generate cards for each project -->
      {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-1">
          {% for project in sorted_projects %}
            {% include projects_horizontal.html %}
          {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="grid">
          {% for project in sorted_projects %}
            {% include projects.html %}
          {% endfor %}
        </div>
      {% endif %}
    {% endfor %}

  {% else %}
  <!-- Display projects without categories -->
    {% assign sorted_projects = site.projects | sort: "importance" %}
    <!-- Generate cards for each project -->
    {% if page.horizontal %}
      <div class="container">
        <div class="row row-cols-2">
        {% for project in sorted_projects %}
          {% include projects_horizontal.html %}
        {% endfor %}
        </div>
      </div>
    {% else %}
      <div class="grid">
        {% for project in sorted_projects %}
          {% include projects.html %}
        {% endfor %}
      </div>
    {% endif %}

  {% endif %}

</div>

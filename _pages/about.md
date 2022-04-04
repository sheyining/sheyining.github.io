---
layout: about
title: About
permalink: /
description: <a href="https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/team/">Human-Cyber-Physical Systems Lab</a> • <a href="https://www.shanghaitech.edu.cn/eng/">ShanghaiTech University</a>

profile:
  align: right
  image: syn-photo1.jpg
  # address: >
  #   <p>SIST 3-403B</p>
  #   <p>393 Middle Huaxia Road</p>
  #   <p>Pudong, Shanghai 201210</p>

news: false  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: true  # includes social icons at the bottom of the page

display_categories: [HCPS Lab, Mars]
horizontal: true
---

I am a senior undergraduate student at <a href="https://www.shanghaitech.edu.cn/eng/">ShanghaiTech University</a> major in Computer Science. Currently, I am an undergraduate research assistant in <a href="https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/team/">Human-Cyber-Physical Systems Lab</a>, advised by Prof. <a href="https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/">Zhihao Jiang</a>.

My current research focuses on developing model-based intelligent assistance systems in Transportation and Healthcare. I realize that research over reliable and intelligent software systems is essential for the next evolution of modern life. Recently, I'm applying for **Ph.D.** position in *Software Engineering* and *Artificial Intelligence* for **Fall 2022**. My research interests includes *Cyber-Physical Systems*, *Formal Methods*, and *Machine Learning*!

Before joining HCPS Lab, I was a member of <a href="https://vic.shanghaitech.edu.cn/">Visual & Data Intelligence Center</a>, advised by Prof. [Jingyi Yu](https://vic.shanghaitech.edu.cn/vrvc/en/people/jingyi-yu/) and Prof. [Lan Xu](http://xu-lan.com/). My work primarily aimed at solving problems in *2D Face Reconstruction* by using the techniques from machine learning and computer graphics.

Please feel free to contact me through [email](mailto:sheyining@live.com) if you have any questions.



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

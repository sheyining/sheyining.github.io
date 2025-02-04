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

display_categories: [SoDA]
horizontal: true
---

I am a second-year PhD student in Software Engineering at [Software and Societal Systems Department](https://s3d.cmu.edu/), [Carnegie Mellon University](https://www.cmu.edu/), advised by [Dr. Eunsuk Kang](https://eskang.github.io/). My research interests are mainly in the intersection of *Software Engineering* and *Artificial Intelligence*. I want to leverage SE techniques to help design safe and reliable AI systems, as well as investigate new applications of AI for software development.

My recent work focuses on analyzing long-term fairness issuesd in ML-enabled systems, and utilizing LLM to extract formal model from source code.

Before joining CMU, I received my bachelor of Engineering in Computer Science at [ShanghaiTech University](https://www.shanghaitech.edu.cn/eng/), where I worked with [Dr. Zhihao Jiang](https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/) at [Human-Cyber-Physical Systems Lab](https://faculty.sist.shanghaitech.edu.cn/faculty/jiangzhh/team/).

If you are interested in my research and collaboration, please feel free to reach out.



<br/>
<br/>
<br/>
<br/>

## Selected Projects
<div class="projects">
  {% if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized projects -->
    {% for category in page.display_categories %}
      <!-- <h2 class="category">{{ category }}</h2> -->
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

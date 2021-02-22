---
layout: page
title: Blog
nav: blog
permalink: /blog/
topics: [SafeRL, LfD, Sim2Real, Misc]
---

{% for t in page.topics %}
  <div class="row m-0 p-0" style="border-top: 1px solid #ddd; flex-direction: row-reverse;">
    <div class="col-sm-1 mt-2 p-0 pr-1">
      <h3 class="bibliography-year">{{t}}</h3>
    </div>
    <div class="col-sm-11 p-0">
          {% assign sorted_posts = site.blog | sort: "date" | reverse %}
          {% for post in sorted_posts %}
            {% if post.topic == t %}
              <a href="{{ post.url }}"><h5 class="title mb-0"> {{post.title}}</h5></a>
              <span class="badge font-weight-bold red align-middle" style="width: 75px;">
                {{ post.venue}}
              </span>
              <a>{{ post.description}}</a>
              <br>
              <br>
            {% endif %}
          {% endfor %}
    </div>
  </div>
{% endfor %}

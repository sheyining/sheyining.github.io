---
layout: post
title: Blog
nav: blog
permalink: /blog/
---

<ul>
  {% assign sorted_posts = site.blog | sort: "date" | reverse %}
  {% for post in sorted_posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title}}</a>
    </li>
  {% endfor %}
</ul>
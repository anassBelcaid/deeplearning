---
layout: page
title: Students Projects
permalink: /project/
---


<h1>{{ page.title }}</h1>
<div class="project-grid">
  {% for project in site.data.projects %}
  <div class="project-cell">
    <a href="{{ project.report }}">
      <img src="{{ project.thumbnail }}" alt="{{ project.name }} Thumbnail">
    </a>
    <h3>{{ project.name }}</h3>
    <p>Students: {{ project.students | join: ', ' }}</p>
    <ul>
      <li><a href="{{ project.report }}">Report</a></li>
      {% if project.presentation1 %}
      <li><a href="{{ project.presentation1 }}">Presentation 1</a></li>
      {% endif %}
      {% if project.presentation2 %}
      <li><a href="{{ project.presentation2 }}">Presentation 2</a></li>
      {% endif %}
    </ul>
  </div>
  {% endfor %}
</div>

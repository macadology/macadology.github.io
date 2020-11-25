---
title: "Food Fermentation"
excerpt: "Food Fermentation"
sitemap: false
---
# Food Fermentation - Figures
{% for image in site.static_files %}
{% if image.path contains 'food_fermentation/' %}
[{{ image.path }}]({{ image.path }})
{% endif %}
{% endfor %}

---
title: "Food Fermentation"
excerpt: "Food Fermentation"
sitemap: false
---
# Plotly Figures
{% for image in site.static_files %}
{% if image.path contains 'food_fermentation/' %}
[{{ image.basename }}]({{ image.path }})
{% endif %}
{% endfor %}

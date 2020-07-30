---
title: "How does the value of a HDB flat depreciate over time? Part 1"
date: 2020-07-30
year: 2020
monthday: 07-30

categories:
  - opinion
tags:
  - HDB
  - resale
  - depreciation
  - bala
header:
  image: /assets/images/2020/07-30/the-pinnacle-2640724_1920.jpg
  teaser: /assets/images/2020/07-30/the-pinnacle-2640724_1920.jpg
  caption: "Photo by ScribblingGeek on pixabay"
---
 {% include addon_scripts.html %}
Suppose you buy a resale apartment with **80 years** of lease remaining today. 10 years later, the apartment appreciates by 25% or ~2.5% per year. Ignoring potential rental yield, 2.5% is a respectable rate of return especially since the apartment is also functioning as a living space for the duration.

Now, suppose you buy a second apartment with **99 years** of lease at the same time, all else being equal (Same location, same sqft etc.) 10 years later, its value appreciates by 100%.

Which apartment is the better investment? Why is there a difference in the annual rate of return? How does depreciation come into play?

# Appreciation vs depreciation
To reconcile how a depreciating asset could grow in value, it is important to distinguish the factors that cause properties to appreciate and/or depreciate.

| Depreciation           | Appreciation                |
|------------------------|-----------------------------|
| Decaying lease         | Increase demand for housing |
| Wear and tear          | Decrease supply             |
| Low demand for housing | Improvement of amenities    |
| Large supply shock     | Land value appreciation     |
|                        | Others                      |

Appreciation and depreciation are not mutually exclusive. As long as the magnitude of appreciation is much larger than depreciation, the value of the property should increase over time. If we only had information on a single leasehold apartment appreciating over time and no way to compare it with other similar apartments, we may wrongly conclude that asset depreciation is a myth.

Nevertheless, the effects of depreciation must be present because both properties in our scenario did not experience the same level of appreciation. To account for the difference, we classify the factors into ones that affect the two apartments differentially and others that affect them equally.

| Differential Factors                                                          | Common Factors       |
|--------------------------------------------------------------------|------------|
| Decaying lease                                                     | Amenities  |
| Wear and tear                                                      | Land Value |
| Differential supply and demand for properties with different lease | Others     |

Looking at the columns, the differential factors tend to cause depreciation whereas the common factors tend to cause appreciation. To simplify the discussion, we shall assume that the rate of appreciation, regardless of factors involved, is the same across all apartments with the same features. The rate of depreciation on the other hand varies depending on the remaining lease. Below is a curve showing the rate of depreciation assuming either linear or non-linear depreciation.

{% include_relative 07-30/depreciation.html %}

Notice that in real terms, the loss of value is constant every year assuming linear depreciation regardless of the number of remaining years. A million dollar property of 100 years lease will lose a [real](https://bit.ly/3hUqAXR) value of $10000 in the first year, $10000 in the last year in the 40th year, and $10000 in the last year. If we measured the relative rate of annual depreciation instead, it changes depending on the year. For example, the decrease from year 0 to 1 is ~1% whereas the decrease from year 99 to 100 is 100%. Mathematically,

\\[ \text{Annual change} = \frac{dln(y)}{dx} = \frac{1}{100-x} \\]

The blue line represents the annual depreciation. It starts out at a low rate of 1% and increases gradually until it reaches 100%. The red line represents a constant rate of appreciation of 2%. A crossover happens at year 50, when annual depreciation equals appreciation.

{% include_relative 07-30/annual_change.html %}

If we combine both rates and calculate the change in value over the term of the lease (see figure below), we see an initial rise due to capital appreciation (Double click on the blue line in the legend). However, the rate of depreciation surpasses appreciation at year 50 and starts eroding the value of the property until it eventually decays to zero.

{% include_relative 07-30/depreciation_w_appreciation.html %}

Freehold apartments on the other hand retain their value well because despite the depreciation from wear and tear, they are built on land that generally grows in value.

In summary, HDB properties will depreciate over time as their leases run out. The depreciation is not immediately clear if we were to examine only the historical prices of a property because of confounding factors such as capital appreciation. In fact, we may even arrive at the erroneous conclusion that depreciation is a myth just because prices did not drop.

To calculate the rate of depreciation independent of appreciation, we should instead compare prices between apartments that are similar in every way except for their lease. Given the price of an apartment with 99 years of lease, if we know the price of similar apartments with 90 years, 80 years and so on, we can fit a curve to calculate the rate of depreciation and answer the following questions:
1. Do HDB apartments depreciate linearly or non-linearly?
2. If the depreciation is non-linear, what is the discount rate?

We shall explore the effects of present value discounting in Part 2.

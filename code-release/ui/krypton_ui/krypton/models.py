# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Photo(models.Model):
    title = models.CharField(max_length=255, blank=True)
    file = models.ImageField(upload_to='photos/', blank=True)
    #uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'photo'

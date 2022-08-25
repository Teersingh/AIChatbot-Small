from django.contrib import admin
from import_export.admin import ImportExportModelAdmin

from .models import ProductBrand

# Register your models here.
class BrandAdmin(ImportExportModelAdmin):
    pass



admin.site.register(ProductBrand,BrandAdmin)

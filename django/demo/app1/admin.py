from django.contrib import admin
from app1.models import Person, Order

# Register your models here.

class PersonAdmin(admin.ModelAdmin):
    list_display = ('first_name', 'last_name', 'created_at', 'updated_at')
    list_filter = ('first_name', 'last_name')
    #元组，只有一个元素时加 ","
    #search_fields = ('first_name',)
    search_fields = ('first_name', 'last_name')
    readonly_fields = ('created_at', 'updated_at')


admin.site.register(Person, PersonAdmin)

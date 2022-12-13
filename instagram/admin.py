from django.contrib import admin
from .models import Post, Comment
from django.utils.safestring import mark_safe

# Register your models here.
# admin.site.register(Post)

@admin.register(Post) # @ Wrapping
class PostAdmin(admin.ModelAdmin):
    list_display = ['id', 'photo_tag','message', 'message_length', 'is_public','created_at', 'updated_at']
    list_display_links = ['message'] # 링크 잡고 싶은 곳 지정
    list_filter = ['created_at', 'is_public']
    search_fields = ['message'] # 검색
    
    def photo_tag(self, post):
        if post.photo:
            return mark_safe(f'<img src = "{post.photo.url}" style="width: 72px;" />')
        return None
    
    def message_length(self, post):
        return len(post.message)
    
@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    pass
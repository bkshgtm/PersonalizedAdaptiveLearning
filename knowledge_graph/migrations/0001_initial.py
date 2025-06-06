# Generated by Django 5.1.7 on 2025-04-02 03:57

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("core", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="KnowledgeGraph",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("description", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("version", models.CharField(max_length=20)),
                ("data", models.JSONField(help_text="Graph structure as JSON")),
                ("is_active", models.BooleanField(default=False)),
                (
                    "created_by",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-updated_at"],
                "unique_together": {("name", "version")},
            },
        ),
        migrations.CreateModel(
            name="GraphEdge",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "relationship_type",
                    models.CharField(
                        choices=[
                            ("prerequisite", "Prerequisite"),
                            ("related", "Related"),
                            ("part_of", "Part Of"),
                            ("next", "Next"),
                        ],
                        max_length=20,
                    ),
                ),
                (
                    "weight",
                    models.FloatField(default=1.0, help_text="Edge weight (0-1)"),
                ),
                (
                    "source_topic",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="outgoing_edges",
                        to="core.topic",
                    ),
                ),
                (
                    "target_topic",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="incoming_edges",
                        to="core.topic",
                    ),
                ),
                (
                    "graph",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="edges",
                        to="knowledge_graph.knowledgegraph",
                    ),
                ),
            ],
            options={
                "ordering": ["graph", "source_topic"],
                "unique_together": {
                    ("graph", "source_topic", "target_topic", "relationship_type")
                },
            },
        ),
    ]

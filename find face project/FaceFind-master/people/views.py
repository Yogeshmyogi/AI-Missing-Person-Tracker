from django.shortcuts import render, get_object_or_404
from django.views.generic import ListView, DetailView, TemplateView
from django.views.generic.edit import CreateView, DeleteView, UpdateView
from django.urls import reverse_lazy
from django.conf import settings
from django.core.mail import send_mail
from django.contrib.auth.mixins import LoginRequiredMixin

from people.models import MissingPerson, ReportedPerson
from .forms import *

import os
import json
import base64
import cv2
import numpy as np
from deepface import DeepFace

# Load config if needed
with open('./config.json', 'r') as f:
    config = json.load(f)

# ---------------- FACE EMBEDDING FUNCTION ---------------- #
def generate_face_embedding(image_path_or_base64):
    if isinstance(image_path_or_base64, str) and image_path_or_base64.startswith('data:image'):
        image_data = base64.b64decode(image_path_or_base64.split(',')[-1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_path_or_base64, str):  # path
        img = image_path_or_base64
    else:
        img = image_path_or_base64

    # Use DeepFace with fallback
    embedding_result = DeepFace.represent(img_path=img, model_name='VGG-Face', enforce_detection=False)
    if not embedding_result:
        raise ValueError("No face found in the image. Please use a clearer frontal photo.")
    
    return embedding_result[0]["embedding"]


# ------------------- MISSING PERSON VIEWS ------------------- #

class MissingPersonListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('index')
    model = MissingPerson
    template_name = 'people/missing_person_list.html'
    context_object_name = "missing_persons"

class MissingPersonToBeApprovedListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('index')
    template_name = 'people/missing_person_list.html'
    context_object_name = 'missing_persons'
    queryset = MissingPerson.objects.filter(is_verified=False)

class MissingPersonWithLeadsListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('index')
    template_name = 'people/missing_person_list.html'
    context_object_name = 'missing_persons'
    queryset = MissingPerson.objects.filter(status='Leads')

class MissingPersonFoundListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('index')
    template_name = 'people/missing_person_list.html'
    context_object_name = 'missing_persons'
    queryset = MissingPerson.objects.filter(status='Found')

class MissingPersonCreateView(CreateView):
    model = MissingPerson
    form_class = MissingPersonCreateForm
    template_name = 'people/create_update_form.html'
    success_url = reverse_lazy('missing_person_form_success')

class MisssingPersonUpdateView(LoginRequiredMixin, UpdateView):
    login_url = reverse_lazy('index')
    model = MissingPerson
    form_class = MissingPersonUpdateForm
    template_name = 'people/create_update_form.html'
    success_url = reverse_lazy('list_missing_person')

class MisssingPersonVerifyView(LoginRequiredMixin, UpdateView):
    login_url = reverse_lazy('index')
    model = MissingPerson
    form_class = MissingPersonVerifyForm
    template_name = 'people/create_update_form.html'
    success_url = reverse_lazy('list_missing_person')

    def form_valid(self, form):
        self.object = form.save(commit=False)

        if form.cleaned_data['is_verified'] and not self.object.face_id:
            try:
                print("Generating face embedding for:", self.object.photo.path)
                embedding = generate_face_embedding(self.object.photo.path)
                self.object.face_id = json.dumps(embedding)
                print("Embedding generated and saved.")

            except Exception as e:
                form.add_error(None, f"Face ID generation failed: {e}")
                return self.form_invalid(form)

        self.object.save()
        return super().form_valid(form)

class MisssingPersonDeleteView(LoginRequiredMixin, DeleteView):
    login_url = reverse_lazy('index')
    model = MissingPerson
    template_name = 'people/delete_form.html'
    success_url = reverse_lazy('list_missing_person')

# ------------------- REPORTED PERSON VIEWS ------------------- #

class ReportedPersonListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('index')
    model = ReportedPerson
    template_name = 'people/reported_person_list.html'
    context_object_name = "reported_persons"

class ReportedPersonToBeApprovedListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('index')
    template_name = 'people/reported_person_list.html'
    context_object_name = 'reported_persons'
    queryset = ReportedPerson.objects.filter(is_verified=False)

class ReportedPersonMatchedListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('index')
    template_name = 'people/reported_person_list.html'
    context_object_name = 'reported_persons'
    queryset = ReportedPerson.objects.filter(is_matched_with_missing_person=True)

class ReportedPersonNotMatchedListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('index')
    template_name = 'people/reported_person_list.html'
    context_object_name = 'reported_persons'
    queryset = ReportedPerson.objects.filter(is_matched_with_missing_person=False, is_verified=True)

class ReportedPersonCreateView(CreateView):
    model = ReportedPerson
    form_class = ReportedPersonCreateForm
    template_name = 'people/reported_create_update_form.html'
    success_url = reverse_lazy('reported_person_form_success')

class ReportedPersonUpdateView(LoginRequiredMixin, UpdateView):
    login_url = reverse_lazy('index')
    model = ReportedPerson
    form_class = ReportedPersonUpdateForm
    template_name = 'people/reported_create_update_form.html'
    success_url = reverse_lazy('list_reported_person')

# Placeholder function: you'll need to implement this using cosine similarity or DeepFace's `find`
def find_match(embedding_to_check, existing_embeddings_json_list):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    embedding_to_check = np.array(json.loads(embedding_to_check)).reshape(1, -1)
    for emb_json in existing_embeddings_json_list:
        existing_emb = np.array(json.loads(emb_json)).reshape(1, -1)
        similarity = cosine_similarity(embedding_to_check, existing_emb)[0][0]
        if similarity > 0.7:
            return emb_json  # return matched embedding
    return None

class ReportedPersonVerifyView(LoginRequiredMixin, UpdateView):
    login_url = reverse_lazy('index')
    model = ReportedPerson
    form_class = ReportedPersonVerifyForm
    template_name = 'people/reported_create_update_form.html'
    success_url = reverse_lazy('list_reported_person')

    def post(self, request, **kwargs):
        form = self.form_class(request.POST)
        if form.is_valid():
            self.object = self.get_object()
            if form.cleaned_data['is_verified'] and not self.object.face_id:
                try:
                    embedding = generate_face_embedding(self.object.photo.path)
                    self.object.face_id = json.dumps(embedding)
                    self.object.save()

                    missing_embeddings = list(MissingPerson.objects.filter(face_id__isnull=False).values_list('face_id', flat=True))
                    match = find_match(self.object.face_id, missing_embeddings)

                    if match:
                        matched_person = MissingPerson.objects.get(face_id=match)
                        matched_person.status = "Leads"
                        matched_person.found_location = self.object.reported_location
                        matched_person.found_time = self.object.created_date
                        matched_person.save()

                        self.object.matched_face_id = match
                        self.object.is_matched_with_missing_person = True
                        self.object.matched_confindence = f"This could be {matched_person.first_name} lost at {matched_person.last_seen} reported by {matched_person.contact_person}."
                        self.object.save()

                except Exception as e:
                    form.add_error(None, f"Face embedding or match failed: {e}")
                    return self.form_invalid(form)

        return super().post(request, **kwargs)

class ReportedPersonDeleteView(LoginRequiredMixin, DeleteView):
    login_url = reverse_lazy('index')
    model = ReportedPerson
    template_name = 'people/delete_form.html'
    success_url = reverse_lazy('list_reported_person')

# ------------------- MISC ------------------- #

class FoundPersonTemplateView(LoginRequiredMixin, TemplateView):
    login_url = reverse_lazy('index')
    template_name = 'people/found_person_details.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['reported_person_details'] = ReportedPerson.objects.filter(matched_face_id=self.kwargs['face_id'])
        context['found_person_details'] = MissingPerson.objects.filter(face_id=self.kwargs['face_id'])
        return context

class MissingPersonFormSuccessView(TemplateView):
    template_name = 'people/missing_person_form_success.html'

class ReportedPersonFormSuccessView(TemplateView):
    template_name = 'people/reported_person_form_success.html'

def SendEmailToContact(object):
    subject = f'We have found {object.first_name}!'
    message = f'Hi {object.contact_person}, {object.first_name} {object.last_name} was reported to be found at {object.found_location} on {object.found_time}.'
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [object.contact_email]
    send_mail(subject, message, email_from, recipient_list)

def missing_person_update_status(request, pk):
    object = get_object_or_404(MissingPerson, pk=pk)
    object.status = "Found"
    SendEmailToContact(object)
    object.is_contacted = True
    object.save()
    context = {'missing_person_object': object}
    return render(request, "people/missing_person_matched.html", context)

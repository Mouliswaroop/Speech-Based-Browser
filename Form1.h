#pragma once

#include<Windows.h>
#include<stdio.h>
#include<sstream>
#include<tchar.h>
#include "HMM\\HMM.h"

namespace Swap_Assistant {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Threading;

	/// <summary>
	/// Summary for Form1
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	public:
		Form1(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Form1()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^  button1;
	protected: 
	private: System::Windows::Forms::Button^  button2;
	private: System::Windows::Forms::Button^  button3;
	private: System::Windows::Forms::Button^  button4;
	private: System::Windows::Forms::Label^  label1;


	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;
	static void live_train()
	{
		hmm_live_training();
	}
	static void assistent()
	{
		hmm_live_testing();
	}
#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(Form1::typeid));
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->button3 = (gcnew System::Windows::Forms::Button());
			this->button4 = (gcnew System::Windows::Forms::Button());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->SuspendLayout();
			// 
			// button1
			// 
			this->button1->BackColor = System::Drawing::Color::White;
			this->button1->BackgroundImage = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"button1.BackgroundImage")));
			this->button1->Font = (gcnew System::Drawing::Font(L"Sitka Text", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->button1->ForeColor = System::Drawing::Color::Black;
			this->button1->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"button1.Image")));
			this->button1->Location = System::Drawing::Point(72, 529);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(213, 125);
			this->button1->TabIndex = 0;
			this->button1->Text = L"Live Training";
			this->button1->UseVisualStyleBackColor = false;
			this->button1->Click += gcnew System::EventHandler(this, &Form1::button1_Click);
			// 
			// button2
			// 
			this->button2->Font = (gcnew System::Drawing::Font(L"Sitka Text", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->button2->ForeColor = System::Drawing::Color::Black;
			this->button2->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"button2.Image")));
			this->button2->Location = System::Drawing::Point(72, 50);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(213, 130);
			this->button2->TabIndex = 1;
			this->button2->Text = L"Assistant";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &Form1::button2_Click);
			// 
			// button3
			// 
			this->button3->Font = (gcnew System::Drawing::Font(L"Sitka Text", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->button3->ForeColor = System::Drawing::Color::Black;
			this->button3->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"button3.Image")));
			this->button3->Location = System::Drawing::Point(1095, 70);
			this->button3->Name = L"button3";
			this->button3->Size = System::Drawing::Size(234, 136);
			this->button3->TabIndex = 2;
			this->button3->Text = L"About";
			this->button3->UseVisualStyleBackColor = true;
			this->button3->Click += gcnew System::EventHandler(this, &Form1::button3_Click);
			// 
			// button4
			// 
			this->button4->BackgroundImage = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"button4.BackgroundImage")));
			this->button4->Font = (gcnew System::Drawing::Font(L"Sitka Text", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->button4->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"button4.Image")));
			this->button4->Location = System::Drawing::Point(1098, 518);
			this->button4->Name = L"button4";
			this->button4->Size = System::Drawing::Size(234, 128);
			this->button4->TabIndex = 3;
			this->button4->Text = L"Exit";
			this->button4->UseVisualStyleBackColor = true;
			this->button4->Click += gcnew System::EventHandler(this, &Form1::button4_Click);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Font = (gcnew System::Drawing::Font(L"Modern No. 20", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->label1->Location = System::Drawing::Point(427, 159);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(157, 21);
			this->label1->TabIndex = 4;
			this->label1->Text = L"Hi!! I am SWAP.";
			this->label1->Click += gcnew System::EventHandler(this, &Form1::label1_Click);
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(7, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::Lavender;
			this->BackgroundImage = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"$this.BackgroundImage")));
			this->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Center;
			this->ClientSize = System::Drawing::Size(1344, 721);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->button4);
			this->Controls->Add(this->button3);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->button1);
			this->Cursor = System::Windows::Forms::Cursors::Default;
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->ForeColor = System::Drawing::Color::Black;
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->Name = L"Form1";
			this->Text = L"Voice Assistent";
			this->Load += gcnew System::EventHandler(this, &Form1::Form1_Load);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {
				 try
			 {
				 System::Media::SoundPlayer^ player = gcnew System::Media::SoundPlayer();
				 player->SoundLocation = "instructions\\training.wav";
				 player->Load();
				 player->PlaySync();

			 }
			 catch( Win32Exception^ ex)
			 {
				 MessageBox::Show(ex->Message);
			 }
			 Thread^ thread = gcnew Thread(gcnew ThreadStart(&live_train));
			 thread->Start();
			 
			 }
	private: System::Void Form1_Load(System::Object^  sender, System::EventArgs^  e) {
				 
				 
			 }
private: System::Void button2_Click(System::Object^  sender, System::EventArgs^  e) {
			 try
			 {
				 System::Media::SoundPlayer^ player = gcnew System::Media::SoundPlayer();
				 player->SoundLocation = "instructions\\assistent.wav";
				 player->Load();
				 player->PlaySync();

			 }
			 catch( Win32Exception^ ex)
			 {
				 MessageBox::Show(ex->Message);
			 }
			 Thread^ thread = gcnew Thread(gcnew ThreadStart(&assistent));
			 thread->Start();
			
			 }
private: System::Void button3_Click(System::Object^  sender, System::EventArgs^  e) {
			  try
			 {
				 System::Media::SoundPlayer^ player = gcnew System::Media::SoundPlayer();
				 player->SoundLocation = "instructions\\about.wav";
				 player->Load();
				 player->PlaySync();

			 }
			 catch( Win32Exception^ ex)
			 {
				 MessageBox::Show(ex->Message);
			 }

			 }
private: System::Void button4_Click(System::Object^  sender, System::EventArgs^  e) {
			 try
			 {
				 System::Media::SoundPlayer^ player = gcnew System::Media::SoundPlayer();
				 player->SoundLocation = "instructions\\signing_off.wav";
				 player->Load();
				 player->PlaySync();

			 }
			 catch( Win32Exception^ ex)
			 {
				 MessageBox::Show(ex->Message);
			 }
			 Application::Exit();
		 }
private: System::Void label1_Click(System::Object^  sender, System::EventArgs^  e) {

		 }
};
}

